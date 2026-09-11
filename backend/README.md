# AetherCheck Stream Core (backend)

The server that sits between the Chrome extension and the two detection engines.
It terminates the WebSocket, decodes PCM, drives the physics + intent engines,
fuses their output into one 0–100 risk score, and streams it back for the HUD.

## Run

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health
```

`/health` tells you which engines actually loaded. **The server boots and streams
even when torch / silero-vad / faster-whisper are missing** — those engines fall
back to null implementations and report themselves offline, so nobody is blocked
waiting on a 2 GB install.

## Test

```bash
cd backend && pytest test_stream_core.py -v
```

18 tests, no ML dependencies required.

## Wire protocol

### Client → server, `ws://localhost:8000/ws/stream`

One JSON object per audio chunk. This is exactly what `extension/offscreen.js`
and `demo_assets/mock_agent.py` already send:

```json
{
  "source": "remote",
  "timestamp_ms": 1731000000000,
  "audio_data": "<base64 of 16 kHz mono 16-bit little-endian PCM>"
}
```

`source` is `"remote"` (the caller, from tab capture) or `"local"` (your mic).
Malformed frames get an `{"type": "error"}` reply and the socket stays open.

### Server → client

On connect:

```json
{"type": "session_start", "session_id": "a1b2c3d4",
 "engines": {"physics": {"available": true, "detail": "ok"},
             "intent":  {"available": false, "detail": "..."}},
 "alert_threshold": 75.0}
```

Then, throttled to ~4/s (an `alert` always goes through immediately):

```json
{
  "type": "risk_update",
  "risk": 82.4,
  "band": "critical",
  "alert": true,
  "physics": {"score": 91.0, "available": true, "turns": 4,
              "last_latency_ms": 1310.0, "median_latency_ms": 1280.0,
              "yield_failures": 1},
  "intent":  {"score": 71.0, "available": true, "transcript": "..."},
  "weights": {"physics": 0.55, "intent": 0.45},
  "triggers": [
    {"id": "pipeline_delay", "label": "Pipeline Delay",
     "detail": "1310ms response gap (human baseline < 750ms)",
     "source": "physics", "ts_ms": 1731000000000.0}
  ],
  "ts_ms": 1731000000000.0
}
```

### `ws://localhost:8000/ws/hud`

Read-only mirror of every `risk_update`. Use it if the overlay is easier to wire
as its own subscriber than to relay through `chrome.runtime` messaging.

## Notes for the HUD (Dev 4)

- `risk` is the dial, 0–100. `band` is `normal` / `elevated` / `critical` if you
  want colour without re-deriving thresholds.
- `alert` is `true` exactly once per crossing of 75. Fire the Chrome desktop
  notification on that edge, not on every frame above threshold, or you get a
  notification storm.
- `triggers` is ready to render as badge cards — `label` on the badge, `detail`
  as the body. Repeats of the same trigger collapse to the newest, max 8.
- `physics.last_latency_ms` is the live turn-around meter.

## Notes for the intent engine

`engines/registry.py` looks for `IntentEngine`, `ScamIntentEngine`, or
`create_engine` in `engines/intent_engine.py`. Expose any one of them with this
method and the server picks it up with no changes here:

```python
def analyze(self, pcm_bytes: bytes, timestamp_ms: float) -> dict | None:
    """pcm_bytes: ~4s window of 16 kHz mono int16 caller audio.

    Return None for nothing to report, else:
        {"score": 0-100,
         "transcript": "...",
         "triggers": [{"id": "authority_impersonation",
                       "label": "Authority Impersonation",
                       "detail": "matched: CBI"}]}
    """
```

It is called in a worker thread (Whisper is blocking), one window in flight at a
time, and only on windows above −50 dBFS so you never transcribe silence.

## Notes for the physics engine

The stream core consumes the dict from `ingest_pcm` as-is. It scores
`latency_ms` whenever present, and it already handles a
`"flag": "YIELD_FAILURE"` for the barge-in / failure-to-yield metric from
Phase 2 — emit that flag when the caller talks through an interruption and the
score picks it up automatically.

## Scoring

```
Risk = 0.55 × Physics_Score + 0.45 × Intent_Score
```

Physics score is derived in `engines/fusion.py` from turn latencies:

- Below 750 ms scores 0; 2500 ms scores 100; linear between.
- Smoothed across turns (EMA) and scaled by confidence, so **one** long pause
  cannot max the dial — it takes a sustained pattern. This is deliberate: a
  human who pauses to think should not be flagged.
- Each failure-to-yield adds 12 points, capped at 36.

If one engine is offline its weight is redistributed rather than scored as zero,
so a physics-only run can still legitimately reach 100.
