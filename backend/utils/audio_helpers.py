"""Audio transport helpers for the AetherCheck stream core.

Every frame that reaches the backend arrives as JSON over the WebSocket:

    {"source": "remote" | "local", "timestamp_ms": 1731000000000,
     "audio_data": "<base64 of 16 kHz mono 16-bit little-endian PCM>"}

This module owns decoding and validating that wire format so the engines only
ever receive clean PCM. Nothing here imports torch or any ML runtime.
"""

import base64
import binascii
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16
VALID_SOURCES = ("local", "remote")

# The extension sends 4096-sample buffers (~256 ms). Anything wildly larger is
# either a malformed frame or a client trying to flood the engine.
MAX_FRAME_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 5  # 5 seconds


class FrameError(ValueError):
    """Raised when an inbound frame does not match the wire contract."""


@dataclass
class AudioFrame:
    """One decoded chunk of PCM from a single channel."""

    source: str
    timestamp_ms: float
    pcm: bytes

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // BYTES_PER_SAMPLE

    @property
    def duration_ms(self) -> float:
        return self.num_samples / SAMPLE_RATE * 1000.0


def decode_base64_pcm(audio_data: str) -> bytes:
    """Decode a base64 payload into raw 16-bit PCM bytes."""
    if not isinstance(audio_data, str):
        raise FrameError("audio_data must be a base64 string")

    try:
        pcm = base64.b64decode(audio_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameError(f"audio_data is not valid base64: {exc}") from exc

    if len(pcm) % BYTES_PER_SAMPLE != 0:
        raise FrameError("PCM payload is not 16-bit aligned (odd byte count)")
    if len(pcm) > MAX_FRAME_BYTES:
        raise FrameError(f"frame too large: {len(pcm)} bytes > {MAX_FRAME_BYTES}")

    return pcm


def parse_frame(message: dict) -> AudioFrame:
    """Validate a decoded JSON message and return an AudioFrame."""
    if not isinstance(message, dict):
        raise FrameError("frame must be a JSON object")

    source = message.get("source")
    if source not in VALID_SOURCES:
        raise FrameError(f"source must be one of {VALID_SOURCES}, got {source!r}")

    timestamp = message.get("timestamp_ms")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        raise FrameError("timestamp_ms must be a number (epoch milliseconds)")

    pcm = decode_base64_pcm(message.get("audio_data"))
    return AudioFrame(source=source, timestamp_ms=float(timestamp), pcm=pcm)


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to the float32 [-1, 1] range the VAD expects."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def rms_dbfs(pcm: bytes) -> float:
    """Loudness of a frame in dBFS. Used to skip silent frames before ASR."""
    samples = pcm_to_float32(pcm)
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))


class AudioRingBuffer:
    """Bounded byte buffer that accumulates one channel for windowed ASR.

    The physics engine consumes audio frame-by-frame, but the intent engine
    needs a few seconds of speech at a time. This collects remote audio until
    `window_seconds` is available, hands it over, and keeps a short overlap so a
    trigger phrase straddling a window boundary is not lost.
    """

    def __init__(self, window_seconds: float = 4.0, overlap_seconds: float = 0.5):
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be smaller than window_seconds")
        self.window_bytes = int(window_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        self.overlap_bytes = int(overlap_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        self._buf = bytearray()

    def append(self, pcm: bytes) -> None:
        self._buf.extend(pcm)

    @property
    def is_ready(self) -> bool:
        return len(self._buf) >= self.window_bytes

    @property
    def duration_ms(self) -> float:
        return len(self._buf) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000.0

    def take_window(self) -> bytes:
        """Return the accumulated window, retaining `overlap_bytes` for the next."""
        window = bytes(self._buf)
        self._buf = bytearray(window[-self.overlap_bytes:]) if self.overlap_bytes else bytearray()
        return window

    def drain(self) -> bytes:
        """Return everything buffered and reset. Used when a call ends."""
        window = bytes(self._buf)
        self._buf = bytearray()
        return window
