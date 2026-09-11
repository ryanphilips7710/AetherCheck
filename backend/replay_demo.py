"""Dual-channel replay harness for the stream core.

`demo_assets/mock_agent.py` streams one file on the `remote` channel only. That
exercises the transport, but it can never produce a latency reading: the physics
engine measures the gap between the *local* speaker stopping and the *remote*
speaker starting, so with no local channel there is no turn to measure.

This replays a scripted two-party call so the whole pipeline (VAD -> latency ->
fusion -> risk) can be verified end to end. Two details matter for the reading to
mean anything:

* **Both channels advance on one shared clock.** A real call is full duplex: at
  every tick both the mic and the tab produce audio. Whoever is not speaking
  sends silence. If you instead send one channel and then the other, the clock
  you stamp frames with no longer matches wall time and the measured gap is
  whatever your send loop happened to do.
* **Only voiced audio is used for turns.** The demo WAVs are up to 29% silence
  (Natural_Conversation.wav opens with 3.3 s of it). Slice them blindly and the
  "response gap" you measure is the file's own silence, not the gap you
  scripted. `voiced_segments()` strips that out first.

Usage:

    # human call: 250 ms gaps, should stay green
    python replay_demo.py ../demo_assets/test_scenarios/Natural_Conversation.wav --gap 250

    # AI agent: 1.4 s of VAD + LLM + TTS every turn, should go red
    python replay_demo.py ../demo_assets/test_scenarios/digital_arrest_attack.wav --gap 1400

Pass --fast to skip real-time pacing.
"""

import argparse
import asyncio
import base64
import json
import wave

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4096                      # ~256 ms, matching the extension
CHUNK_BYTES = CHUNK_SAMPLES * 2
CHUNK_MS = CHUNK_SAMPLES / SAMPLE_RATE * 1000.0
SILENCE = b"\x00" * CHUNK_BYTES
VOICED_FLOOR_DBFS = -45.0


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (1, 2, SAMPLE_RATE):
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit PCM")
        return wf.readframes(wf.getnframes())


def voiced_segments(pcm: bytes, min_chunks: int = 4) -> list:
    """Split PCM into runs of audible chunks, dropping the silence between them.

    Energy-based on purpose: this harness must not depend on the VAD it is being
    used to test.
    """
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, current = [], []
    for i in range(0, len(samples) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        frame = samples[i:i + CHUNK_SAMPLES]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        dbfs = 20.0 * np.log10(max(rms, 1e-9))
        raw = pcm[i * 2:(i + CHUNK_SAMPLES) * 2]
        if dbfs >= VOICED_FLOOR_DBFS:
            current.append(raw)
        elif current:
            if len(current) >= min_chunks:
                segments.append(current)
            current = []
    if len(current) >= min_chunks:
        segments.append(current)
    return segments


async def replay(url: str, segments: list, gap_ms: int, turns: int,
                 turn_chunks: int, fast: bool) -> float:
    clock_ms = 0.0
    highest = 0.0
    flat = [chunk for seg in segments for chunk in seg]
    if not flat:
        raise SystemExit("no voiced audio found in that file")
    cursor = 0

    async with websockets.connect(url) as ws:
        hello = json.loads(await ws.recv())
        print(f"[+] session {hello.get('session_id')}")
        print(f"    engines: {hello.get('engines')}\n")

        async def tick(local: bytes, remote: bytes) -> None:
            """One 256 ms slice of a full-duplex call: both channels, one clock."""
            nonlocal clock_ms
            for source, pcm in (("local", local), ("remote", remote)):
                await ws.send(json.dumps({
                    "source": source,
                    "timestamp_ms": clock_ms,
                    "audio_data": base64.b64encode(pcm).decode(),
                }))
            clock_ms += CHUNK_MS
            if not fast:
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)

        async def drain() -> None:
            nonlocal highest
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.01)
                except asyncio.TimeoutError:
                    return
                msg = json.loads(raw)
                if msg.get("type") != "risk_update":
                    continue
                highest = max(highest, msg["risk"])
                gap = msg["physics"]["last_latency_ms"]
                labels = ",".join(t["label"] for t in msg["triggers"]) or "-"
                print(f"    risk={msg['risk']:5.1f} [{msg['band']:>8}] "
                      f"turns={msg['physics']['turns']} "
                      f"last_gap={gap if gap is not None else '-'}ms  {labels}")

        def next_chunks(count: int) -> list:
            nonlocal cursor
            out = [flat[(cursor + i) % len(flat)] for i in range(count)]
            cursor += count
            return out

        gap_ticks = max(1, round(gap_ms / CHUNK_MS))
        for turn in range(1, turns + 1):
            print(f"[turn {turn}] user holds the floor")
            for chunk in next_chunks(turn_chunks):
                await tick(local=chunk, remote=SILENCE)
            await drain()

            print(f"[turn {turn}] {gap_ticks * CHUNK_MS:.0f}ms of silence, then the caller replies")
            for _ in range(gap_ticks):
                await tick(local=SILENCE, remote=SILENCE)
            for chunk in next_chunks(turn_chunks):
                await tick(local=SILENCE, remote=chunk)
            await drain()

        await asyncio.sleep(0.5)   # let any in-flight intent analysis land
        await drain()

    print(f"\n[=] peak risk: {highest:.1f}")
    return highest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="16 kHz mono 16-bit PCM file to use as speech")
    ap.add_argument("--gap", type=int, default=1400,
                    help="ms of silence before the caller replies (default 1400)")
    ap.add_argument("--turns", type=int, default=5, help="turns to replay")
    ap.add_argument("--turn-chunks", type=int, default=6,
                    help="256ms chunks each speaker holds the floor for")
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--fast", action="store_true", help="skip real-time pacing")
    args = ap.parse_args()

    segments = voiced_segments(read_wav(args.wav))
    voiced_s = sum(len(s) for s in segments) * CHUNK_MS / 1000.0
    print(f"[*] {args.wav}: {len(segments)} voiced segments, {voiced_s:.1f}s of speech")
    print(f"[*] replaying {args.turns} turns with a {args.gap}ms response gap\n")

    try:
        asyncio.run(replay(args.url, segments, args.gap, args.turns,
                           args.turn_chunks, args.fast))
    except OSError as exc:
        raise SystemExit(f"[!] cannot reach {args.url}: {exc}\n"
                         f"    start it with: cd backend && uvicorn main:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
