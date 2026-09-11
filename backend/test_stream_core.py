"""Contract tests for the stream core.

These run with no ML dependencies installed: the engines fall back to their null
implementations, which is exactly the case we need to stay green so the rest of
the team can develop against the transport layer.

    cd backend && pytest test_stream_core.py -v
"""

import base64
import json
import math
import struct

import pytest
from fastapi.testclient import TestClient

from engines.fusion import ALERT_THRESHOLD, FusionScorer, latency_points
from main import app
from utils.audio_helpers import AudioRingBuffer, FrameError, parse_frame, rms_dbfs

SAMPLE_RATE = 16000


def tone_pcm(duration_s: float = 0.256, freq: float = 440.0) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return struct.pack(
        f"<{n}h",
        *(int(0.6 * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)),
    )


def frame(source: str, timestamp_ms: float, pcm: bytes = None) -> str:
    return json.dumps({
        "source": source,
        "timestamp_ms": timestamp_ms,
        "audio_data": base64.b64encode(tone_pcm() if pcm is None else pcm).decode(),
    })


# --- wire format ---------------------------------------------------------

def test_parse_frame_accepts_the_documented_shape():
    parsed = parse_frame(json.loads(frame("remote", 1000.0)))
    assert parsed.source == "remote"
    assert parsed.timestamp_ms == 1000.0
    assert parsed.num_samples == int(SAMPLE_RATE * 0.256)
    assert parsed.duration_ms == pytest.approx(256.0)


@pytest.mark.parametrize("message", [
    {"source": "speaker", "timestamp_ms": 1, "audio_data": ""},   # bad channel
    {"source": "remote", "audio_data": ""},                        # no timestamp
    {"source": "remote", "timestamp_ms": 1, "audio_data": "!!!"},  # not base64
    {"source": "remote", "timestamp_ms": 1, "audio_data": "AA=="},  # 1 byte: not int16 aligned
])
def test_parse_frame_rejects_malformed_frames(message):
    with pytest.raises(FrameError):
        parse_frame(message)


def test_rms_dbfs_separates_tone_from_silence():
    assert rms_dbfs(tone_pcm()) > -20.0
    assert rms_dbfs(b"\x00\x00" * 4096) < -100.0


def test_ring_buffer_keeps_overlap_between_windows():
    buf = AudioRingBuffer(window_seconds=1.0, overlap_seconds=0.25)
    assert not buf.is_ready
    buf.append(tone_pcm(duration_s=1.0))
    assert buf.is_ready
    window = buf.take_window()
    assert len(window) == SAMPLE_RATE * 2
    assert buf.duration_ms == pytest.approx(250.0)  # overlap retained


# --- scoring -------------------------------------------------------------

def test_latency_below_the_human_floor_scores_nothing():
    assert latency_points(300.0) == 0.0
    assert latency_points(750.0) == 0.0
    assert latency_points(1800.0) == 100.0
    assert latency_points(3000.0) == 100.0
    assert 0 < latency_points(1250.0) < 100


def test_a_typical_voice_agent_gap_scores_most_of_the_physics_range():
    """1.4s is where real VAD + LLM + TTS pipelines land; it must read hot."""
    assert latency_points(1400.0) > 60.0


def test_one_slow_turn_does_not_max_the_dial():
    """A single long pause is suspicious, not conclusive."""
    scorer = FusionScorer()
    scorer.update_physics({"latency_ms": 1800.0, "flag": "AI_PIPELINE_DELAY_SUSPECTED"})
    assert scorer.risk < ALERT_THRESHOLD


def test_a_sustained_pattern_of_pipeline_delay_raises_the_alarm():
    scorer = FusionScorer()
    for _ in range(4):
        scorer.update_physics({"latency_ms": 1900.0, "flag": "AI_PIPELINE_DELAY_SUSPECTED"})
    snap = scorer.snapshot()
    assert snap["risk"] >= ALERT_THRESHOLD
    assert snap["band"] == "critical"
    assert snap["alert"] is True
    assert any(t["id"] == "pipeline_delay" for t in snap["triggers"])


def test_human_paced_conversation_stays_green():
    scorer = FusionScorer()
    for gap in (180.0, 240.0, 120.0, 310.0, 90.0):
        scorer.update_physics({"latency_ms": gap, "flag": "NORMAL"})
    snap = scorer.snapshot()
    assert snap["risk"] == 0.0
    assert snap["band"] == "normal"
    assert snap["triggers"] == []


def test_missing_intent_engine_redistributes_its_weight():
    """With intent offline, physics alone must still be able to reach 100."""
    scorer = FusionScorer()
    for _ in range(5):
        scorer.update_physics({"latency_ms": 3000.0, "flag": "AI_PIPELINE_DELAY_SUSPECTED"})
    assert scorer.intent_available is False
    assert scorer.risk == pytest.approx(scorer.physics.score)


def test_fusion_applies_the_documented_weights_when_both_engines_report():
    scorer = FusionScorer()
    for _ in range(5):
        scorer.update_physics({"latency_ms": 3000.0, "flag": "AI_PIPELINE_DELAY_SUSPECTED"})
    scorer.update_intent({
        "score": 40.0,
        "transcript": "this is a digital arrest, do not hang up",
        "triggers": [{"id": "authority_impersonation", "label": "Authority Impersonation",
                      "detail": "matched: digital arrest"}],
    })
    expected = 0.55 * scorer.physics.score + 0.45 * 40.0
    assert scorer.risk == pytest.approx(expected)
    assert any(t["source"] == "intent" for t in scorer.snapshot()["triggers"])


def test_repeat_triggers_collapse_to_the_newest():
    scorer = FusionScorer()
    for _ in range(6):
        scorer.update_physics({"latency_ms": 1800.0, "flag": "AI_PIPELINE_DELAY_SUSPECTED"})
    ids = [t["id"] for t in scorer.snapshot()["triggers"]]
    assert ids.count("pipeline_delay") == 1


def test_yield_failure_is_scored_and_surfaced():
    scorer = FusionScorer()
    scorer.update_physics({"latency_ms": None, "flag": "YIELD_FAILURE"})
    snap = scorer.snapshot()
    assert snap["physics"]["yield_failures"] == 1
    assert snap["risk"] > 0
    assert any(t["id"] == "yield_failure" for t in snap["triggers"])


# --- transport end to end ------------------------------------------------

def test_health_reports_engine_availability():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert set(body["engines"]) == {"physics", "intent"}


def test_websocket_handshake_and_streaming():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "session_start"
            assert "session_id" in hello

            ws.send_text(frame("local", 1000.0))
            ws.send_text(frame("remote", 3300.0))
            update = ws.receive_json()
            assert update["type"] == "risk_update"
            assert 0.0 <= update["risk"] <= 100.0
            assert update["band"] in ("normal", "elevated", "critical")
            assert update["weights"] == {"physics": 0.55, "intent": 0.45}


def test_websocket_reports_a_bad_frame_without_dropping_the_connection():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # session_start
            ws.send_text(json.dumps({"source": "moon", "timestamp_ms": 1, "audio_data": ""}))
            err = ws.receive_json()
            assert err["type"] == "error"

            ws.send_text(frame("remote", 2000.0))
            assert ws.receive_json()["type"] == "risk_update"
