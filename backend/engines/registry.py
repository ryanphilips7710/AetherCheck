"""Engine loading with graceful degradation.

The physics engine needs torch + silero-vad and the intent engine needs
faster-whisper. On a laptop that has not finished installing them (or before a
teammate has landed their engine) we still want the server to boot, accept
WebSocket frames, and drive the HUD. So each engine is loaded behind a try and
falls back to a null implementation that reports itself as unavailable.

Check `/health` to see which engines actually came up.
"""

import logging
from typing import Optional, Protocol

log = logging.getLogger("aethercheck.registry")


class PhysicsEngineLike(Protocol):
    def ingest_pcm(self, source: str, pcm_bytes: bytes, timestamp_ms: float) -> dict:
        ...


class IntentEngineLike(Protocol):
    def analyze(self, pcm_bytes: bytes, timestamp_ms: float) -> Optional[dict]:
        ...


class NullPhysicsEngine:
    """Stands in when torch / silero-vad is not importable."""

    available = False
    reason = "silero-vad or torch not installed"

    def ingest_pcm(self, source: str, pcm_bytes: bytes, timestamp_ms: float) -> dict:
        return {"source": source, "event": None, "latency_ms": None, "flag": "ENGINE_OFFLINE"}


class NullIntentEngine:
    """Stands in until the intent engine lands (or faster-whisper is missing).

    The contract the real engine must satisfy:

        analyze(pcm_bytes, timestamp_ms) -> None | {
            "score": float,          # 0-100 coercion / impersonation intent
            "transcript": str,       # text for this window
            "triggers": [            # zero or more matched phrases
                {"id": "authority_impersonation",
                 "label": "Authority Impersonation",
                 "detail": "matched: CBI"},
            ],
        }

    `pcm_bytes` is 16 kHz mono 16-bit little-endian PCM, roughly a 4 s window.
    Returning None means "nothing to report for this window".
    """

    available = False
    reason = "intent engine not wired up yet"

    def analyze(self, pcm_bytes: bytes, timestamp_ms: float) -> Optional[dict]:
        return None


def load_physics_engine() -> PhysicsEngineLike:
    try:
        from .physics_engine import ConversationalPhysicsEngine

        engine = ConversationalPhysicsEngine()
        engine.available = True
        engine.reason = "ok"
        log.info("Physics engine (Silero VAD) loaded.")
        return engine
    except Exception as exc:  # ImportError, model download failure, no network
        log.warning("Physics engine unavailable, using null engine: %s", exc)
        null = NullPhysicsEngine()
        null.reason = f"{type(exc).__name__}: {exc}"
        return null


def load_intent_engine() -> IntentEngineLike:
    try:
        from . import intent_engine as intent_module
    except Exception as exc:
        log.warning("Intent engine module not importable: %s", exc)
        null = NullIntentEngine()
        null.reason = f"{type(exc).__name__}: {exc}"
        return null

    # The intent engine owner may expose either a class or a factory; accept
    # the common spellings so this does not need editing when it lands.
    for attr in ("IntentEngine", "ScamIntentEngine", "create_engine"):
        factory = getattr(intent_module, attr, None)
        if factory is None:
            continue
        try:
            engine = factory()
        except Exception as exc:
            log.warning("Intent engine %s failed to construct: %s", attr, exc)
            null = NullIntentEngine()
            null.reason = f"{type(exc).__name__}: {exc}"
            return null
        if not hasattr(engine, "analyze"):
            log.warning("Intent engine %s has no analyze(); using null engine.", attr)
            continue
        engine.available = True
        engine.reason = "ok"
        log.info("Intent engine loaded via %s.", attr)
        return engine

    log.info("No intent engine found in intent_engine.py yet; using null engine.")
    return NullIntentEngine()
