"""Dual-channel replay harness for the stream core.

`demo_assets/mock_agent.py` streams one file on the `remote` channel only. That
exercises the transport, but it can never produce a latency reading: the physics
engine measures the gap between the *local* speaker stopping and the *remote*
speaker starting, so with no local channel there is no turn to measure.

This replays a scripted two-party call over both channels so the whole pipeline
(VAD -> latency -> fusion -> risk) can be verified end to end:

    # a human-paced call: short gaps, score should stay green
    python replay_demo.py ../demo_assets/test_scenarios/Natural_Conversation.wav --gap 250

    # an AI agent: 1.4s of VAD + LLM + TTS on every turn, score should go red
    python replay_demo.py ../demo_assets/test_scenarios/digital_arrest_attack.wav --gap 1400

Pass --fast to skip real-time pacing (useful in CI).
"""

import argparse
import asyncio
import base64
import json
import wave

import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4096          # ~256 ms, matching the extension buffer size
CHUNK_BYTES = CHUNK_SAMPLES * 2
SILENCE = b"\x00" * CHUNK_BYTES


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (1, 2, SAMPLE_RATE):
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit PCM")
        return wf.readframes(wf.getnframes())


def chunks(pcm: bytes):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


async def replay(url: str, caller_pcm: bytes, gap_ms: int, turns: int, fast: bool) -> int:
    clock_ms = 0.0
    highest = 0.0
    per_turn = max(1, len(list(chunks(caller_pcm))) // turns)
    caller_chunks = list(chunks(caller_pcm))

    async with websockets.connect(url) as ws:
        hello = json.loads(await ws.recv())
        print(f"[+] session {hello.get('session_id')}  engines={hello.get('engines')}\n")

        async def send(source: str, pcm: bytes) -> None:
            nonlocal clock_ms
            await ws.send(json.dumps({
                "source": source,
                "timestamp_ms": clock_ms,
                "audio_data": base64.b64encode(pcm).decode(),
            }))
            clock_ms += CHUNK_SAMPLES / SAMPLE_RATE * 1000.0
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
                latency = msg["physics"]["last_latency_ms"]
                labels = ",".join(t["label"] for t in msg["triggers"]) or "-"
                print(f"    risk={msg['risk']:5.1f} [{msg['band']:>8}] "
                      f"turns={msg['physics']['turns']} "
                      f"last_gap={latency if latency is not None else '-'}ms  {labels}")

        cursor = 0
        for turn in range(1, turns + 1):
            print(f"[turn {turn}] user speaks (local)")
            # One second of the user holding the floor.
            for _ in range(4):
                await send("local", caller_chunks[cursor % len(caller_chunks)])
            await drain()

            print(f"[turn {turn}] silence for {gap_ms}ms, then caller replies (remote)")
            # Silence on both channels advances the shared clock across the gap.
            for _ in range(max(1, round(gap_ms / 256))):
                await send("local", SILENCE)

            for chunk in caller_chunks[cursor:cursor + per_turn]:
                await send("remote", chunk)
            cursor += per_turn
            await drain()

        # Give any in-flight intent analysis a moment to land.
        await asyncio.sleep(0.5)
        await drain()

    print(f"\n[=] peak risk: {highest:.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="16 kHz mono 16-bit PCM file to use as caller audio")
    ap.add_argument("--gap", type=int, default=1400,
                    help="ms of silence before the caller replies (default 1400)")
    ap.add_argument("--turns", type=int, default=4, help="number of turns to replay")
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--fast", action="store_true", help="skip real-time pacing")
    args = ap.parse_args()

    print(f"[*] replaying {args.wav} with a {args.gap}ms response gap "
          f"over {args.turns} turns\n")
    try:
        return asyncio.run(replay(args.url, read_wav(args.wav), args.gap, args.turns, args.fast))
    except OSError as exc:
        raise SystemExit(f"[!] cannot reach {args.url}: {exc}\n"
                         f"    start it with: cd backend && uvicorn main:app --port 8000")


if __name__ == "__main__":
    raise SystemExit(main())
