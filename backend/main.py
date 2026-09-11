"""AetherCheck stream core.

FastAPI server that terminates the extension WebSocket, fans raw PCM out to the
conversational-physics and intent engines, fuses their output into a single
0-100 risk score, and streams that score back for the HUD to render.

Run it:

    cd backend
    uvicorn main:app --reload --port 8000

Wire protocol (client -> server), one JSON object per audio chunk:

    {"source": "remote" | "local",
     "timestamp_ms": 1731000000000,
     "audio_data": "<base64 16 kHz mono 16-bit PCM>"}

Wire protocol (server -> client):

    {"type": "session_start", "session_id": "...", "engines": {...}}
    {"type": "risk_update", "risk": 82.4, "band": "critical", "triggers": [...]}
    {"type": "error", "detail": "..."}
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engines.fusion import ALERT_THRESHOLD, FusionScorer
from engines.registry import load_intent_engine, load_physics_engine
from utils.audio_helpers import AudioRingBuffer, FrameError, rms_dbfs, parse_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aethercheck")

# Do not push an identical snapshot more often than this; the extension sends a
# frame every ~256 ms per channel and the dial does not need to redraw faster.
MIN_BROADCAST_INTERVAL_S = 0.25

# Windows quieter than this are almost certainly silence or line noise; running
# Whisper on them wastes the GPU and produces hallucinated text.
ASR_SILENCE_FLOOR_DBFS = -50.0


class HudHub:
    """Read-only subscribers (the content-script overlay) that mirror scores."""

    def __init__(self) -> None:
        self._sockets: Set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._sockets.add(ws)

    def discard(self, ws: WebSocket) -> None:
        self._sockets.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        if not self._sockets:
            return
        message = json.dumps(payload)
        dead = []
        for ws in list(self._sockets):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._sockets.discard(ws)


hud_hub = HudHub()


class StreamSession:
    """One live call: its engines, its buffers, its score."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.physics = load_physics_engine()
        self.intent = load_intent_engine()
        self.fusion = FusionScorer()
        self.remote_buffer = AudioRingBuffer(window_seconds=4.0, overlap_seconds=0.5)
        self.frames = 0
        self.started_at = time.time()
        self._asr_task: Optional[asyncio.Task] = None
        self._last_broadcast = 0.0
        self._last_payload_key: Optional[tuple] = None

    @property
    def engine_status(self) -> dict:
        return {
            "physics": {
                "available": getattr(self.physics, "available", False),
                "detail": getattr(self.physics, "reason", "unknown"),
            },
            "intent": {
                "available": getattr(self.intent, "available", False),
                "detail": getattr(self.intent, "reason", "unknown"),
            },
        }

    def ingest(self, frame) -> None:
        """Feed one frame to the physics engine (cheap, synchronous)."""
        self.frames += 1
        try:
            result = self.physics.ingest_pcm(frame.source, frame.pcm, frame.timestamp_ms)
        except Exception as exc:
            log.exception("Physics engine raised on a %s frame: %s", frame.source, exc)
            return
        self.fusion.update_physics(result)

        if frame.source == "remote":
            self.remote_buffer.append(frame.pcm)

    def maybe_start_asr(self) -> None:
        """Kick off intent analysis when a full window of caller audio is ready.

        Whisper is blocking and slower than real time on CPU, so it runs in a
        worker thread and only one window is ever in flight. If analysis is
        still running when the next window fills, we keep buffering rather than
        queueing work we can never catch up on.
        """
        if not self.remote_buffer.is_ready:
            return
        if self._asr_task is not None and not self._asr_task.done():
            return

        window = self.remote_buffer.take_window()
        if rms_dbfs(window) < ASR_SILENCE_FLOOR_DBFS:
            return

        self._asr_task = asyncio.create_task(self._run_asr(window))

    async def _run_asr(self, window: bytes) -> None:
        try:
            result = await asyncio.to_thread(
                self.intent.analyze, window, time.time() * 1000.0
            )
        except Exception as exc:
            log.exception("Intent engine raised: %s", exc)
            return
        if result:
            self.fusion.update_intent(result)

    def should_broadcast(self, payload: dict) -> bool:
        """Rate-limit, but never swallow the moment the score crosses the alarm."""
        if payload.get("alert"):
            return True
        now = time.time()
        key = (
            payload["risk"],
            payload["band"],
            len(payload["triggers"]),
            payload["physics"]["turns"],
        )
        if key == self._last_payload_key and now - self._last_broadcast < 2.0:
            return False
        if now - self._last_broadcast < MIN_BROADCAST_INTERVAL_S:
            return False
        self._last_broadcast = now
        self._last_payload_key = key
        return True

    async def close(self) -> None:
        if self._asr_task is not None and not self._asr_task.done():
            self._asr_task.cancel()
            try:
                await self._asr_task
            except (asyncio.CancelledError, Exception):
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models once at boot so the first call does not pay the download.
    log.info("Warming engines...")
    warm = StreamSession()
    app.state.engine_status = warm.engine_status
    for name, status in warm.engine_status.items():
        state = "ready" if status["available"] else f"OFFLINE ({status['detail']})"
        log.info("  %s engine: %s", name, state)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="AetherCheck Stream Core",
    description="Real-time turn-physics + intent fusion for AI voice-scam detection",
    version="1.0.0",
    lifespan=lifespan,
)

# The extension talks to us from an offscreen document with a chrome-extension://
# origin, and the demo scripts from localhost. This server is local-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "service": "AetherCheck Stream Core",
        "stream_endpoint": "/ws/stream",
        "hud_endpoint": "/ws/hud",
        "alert_threshold": ALERT_THRESHOLD,
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engines": getattr(app.state, "engine_status", {}),
        "hint": "An offline engine still streams; its weight is redistributed.",
    }


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    session = StreamSession()
    log.info("[%s] Stream opened.", session.id)

    await ws.send_json({
        "type": "session_start",
        "session_id": session.id,
        "engines": session.engine_status,
        "alert_threshold": ALERT_THRESHOLD,
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame = parse_frame(json.loads(raw))
            except (json.JSONDecodeError, FrameError) as exc:
                log.warning("[%s] Rejected frame: %s", session.id, exc)
                await ws.send_json({"type": "error", "detail": str(exc)})
                continue

            session.ingest(frame)
            session.maybe_start_asr()

            payload = session.fusion.snapshot()
            payload["session_id"] = session.id
            if session.should_broadcast(payload):
                await ws.send_json(payload)
                await hud_hub.broadcast(payload)
                if payload["alert"]:
                    log.warning(
                        "[%s] ALERT risk=%s triggers=%s",
                        session.id, payload["risk"],
                        [t["label"] for t in payload["triggers"]],
                    )

    except WebSocketDisconnect:
        log.info(
            "[%s] Stream closed after %d frames in %.1fs (final risk %.1f).",
            session.id, session.frames, time.time() - session.started_at,
            session.fusion.risk,
        )
    except Exception as exc:
        log.exception("[%s] Stream failed: %s", session.id, exc)
    finally:
        await session.close()


@app.websocket("/ws/hud")
async def hud(ws: WebSocket) -> None:
    """Read-only mirror of every risk update, for the overlay to subscribe to."""
    await ws.accept()
    hud_hub.add(ws)
    try:
        while True:
            # The HUD never sends anything; this just parks until it disconnects.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hud_hub.discard(ws)
