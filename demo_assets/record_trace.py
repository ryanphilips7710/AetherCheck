"""Record a live risk trace off /ws/hud so a demo can be replayed offline.

    python record_trace.py out.json --seconds 60

Subscribes to the stream core's HUD mirror and writes every risk_update with
the millisecond offset it arrived at. The demo player replays that timeline at
the original speed, so what an audience sees is real captured output rather
than a mock-up.
"""

import argparse
import asyncio
import json
import sys
import time

import websockets


async def record(url: str, out: str, seconds: float) -> None:
    frames = []
    start = time.time()
    print(f"[*] recording from {url} for up to {seconds:.0f}s (Ctrl-C to stop early)")
    try:
        async with websockets.connect(url) as ws:
            while time.time() - start < seconds:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if msg.get("type") != "risk_update":
                    continue
                frames.append({"t": round((time.time() - start) * 1000), "msg": msg})
                print(f"\r    {len(frames)} frames, risk={msg['risk']:5.1f}", end="")
    except KeyboardInterrupt:
        pass

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(frames, fh, indent=1)
    peak = max((f["msg"]["risk"] for f in frames), default=0)
    print(f"\n[+] wrote {len(frames)} frames to {out} (peak risk {peak})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--url", default="ws://localhost:8000/ws/hud")
    args = ap.parse_args()
    asyncio.run(record(args.url, args.out, args.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
