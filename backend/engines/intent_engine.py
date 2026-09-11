"""Intent engine: what the caller is actually saying.

Transcribes the caller's channel with faster-whisper and scores the transcript
against the coercion playbook a digital-arrest scam has to follow. The physics
engine answers "is this a machine?"; this answers "is this a scam?". Either
alone produces false positives - a slow human on a bad line, or a legitimate
bank call that mentions a transfer. Together they do not.

Scoring is deliberately rule-based rather than a classifier. The categories
below are the script itself: an impersonated authority, isolation from anyone
who would talk you down, and a financial instruction under time pressure. A
caller who hits three of those is running the script regardless of how the
sentence is phrased, and we can show a judge exactly which phrase fired.

Contract (consumed by engines/registry.py):

    analyze(pcm_bytes, timestamp_ms) -> None | {"score", "transcript", "triggers"}
"""

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger("aethercheck.intent")

# tiny.en keeps a 4 s window under ~400 ms on CPU, which is what we need to stay
# ahead of a live call. Override with AETHERCHECK_WHISPER_MODEL=base.en for
# better accuracy if you have the headroom.
WHISPER_MODEL = os.environ.get("AETHERCHECK_WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE = os.environ.get("AETHERCHECK_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("AETHERCHECK_WHISPER_COMPUTE", "int8")

# Keep the last N characters of the call for display and matching.
TRANSCRIPT_LIMIT = 4000


@dataclass
class Category:
    """One move in the scam script."""

    id: str
    label: str
    weight: float          # points this category contributes when it fires
    patterns: List[str]

    def __post_init__(self) -> None:
        # Word-boundary match so "fir" does not fire inside "first" or "confirm".
        self._regex = [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in self.patterns]

    def match(self, text: str) -> Optional[str]:
        """`text` is expected to be normalise()d already."""
        for rx in self._regex:
            found = rx.search(text)
            if found:
                return found.group(0)
        return None


def normalise(text: str) -> str:
    r"""Flatten transcript punctuation and spacing before matching.

    Whisper transcribes in windows, so a phrase can arrive split across two of
    them and come back as "placed under digital. Arrest. Do not..." - the words
    are all there but a \bdigital arrest\b pattern will not see them. Stripping
    punctuation and collapsing whitespace makes matching depend on the words
    the caller said rather than on where the window boundary happened to land.
    """
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


CATEGORIES = [
    Category(
        id="authority_impersonation",
        label="Authority Impersonation",
        weight=35.0,
        patterns=[
            r"cbi", r"c\.?b\.?i", r"central bureau", r"narcotics", r"customs",
            r"enforcement directorate", r"cyber ?crime", r"cyber cell",
            r"telecom department", r"trai", r"police", r"inspector",
            r"f\.?i\.?r", r"court order", r"arrest warrant", r"warrant",
            r"legal notice", r"income tax department", r"aadhaar", r"aadhar",
        ],
    ),
    Category(
        id="digital_arrest",
        label="Digital Arrest Threat",
        weight=40.0,
        patterns=[
            r"digital arrest", r"under arrest", r"you are being arrested",
            r"do not hang up", r"don'?t hang up", r"stay on the line",
            r"keep the line open", r"remain on video", r"cannot disconnect",
            r"case (?:has been )?registered against", r"non.?bailable",
        ],
    ),
    Category(
        id="isolation_pressure",
        label="Isolation Pressure",
        weight=25.0,
        patterns=[
            r"do not tell", r"don'?t tell", r"not inform", r"tell (?:no ?one|nobody)",
            r"inform any family", r"inform your family", r"do not disconnect",
            r"without informing", r"confidential", r"go to a (?:separate |private )?room",
            r"lock the door", r"be alone", r"stay alone", r"no one (?:should|must) know",
            r"official secrets", r"do not discuss",
        ],
    ),
    Category(
        id="financial_escalation",
        label="Financial Escalation",
        weight=30.0,
        patterns=[
            r"rtgs", r"neft", r"imps", r"wire transfer", r"transfer immediately",
            r"verification fee", r"security deposit", r"safety deposit",
            r"safe account", r"refundable", r"verify your funds", r"verify funds", r"account (?:will be |is )?(?:frozen|blocked|suspended)",
            r"aadhaar (?:is )?blocked", r"o\.?t\.?p", r"one time password",
            r"share (?:your )?(?:card|bank|account) details", r"upi",
            r"bitcoin", r"gift card",
        ],
    ),
]


class IntentEngine:
    """Rolling scam-intent score over the caller's channel."""

    available = False
    reason = "not initialised"

    def __init__(self) -> None:
        self.transcript = ""
        self.matched: dict = {}          # category id -> matched phrase
        self._model = None
        self._model_lock = threading.Lock()
        self._load_failed: Optional[str] = None

        # Load eagerly so /health reports the truth at boot rather than failing
        # halfway through the first call.
        try:
            self._ensure_model()
            self.available = True
            self.reason = f"faster-whisper {WHISPER_MODEL} ({WHISPER_COMPUTE})"
        except Exception as exc:
            self.available = False
            self.reason = f"{type(exc).__name__}: {exc}"
            log.warning("Intent engine could not load Whisper: %s", exc)

    # -- transcription -----------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info("Loading faster-whisper %s on %s (%s)...",
                         WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
                self._model = WhisperModel(
                    WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
                )
        return self._model

    def transcribe(self, pcm_bytes: bytes) -> str:
        model = self._ensure_model()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,                 # greedy: this is a latency-bound path
            # Whisper's own VAD filter needs onnxruntime and duplicates work we
            # already do: main.py gates windows at -50 dBFS before calling us.
            vad_filter=False,
            condition_on_previous_text=False,   # stops hallucination loops
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # -- scoring -----------------------------------------------------------
    def score_text(self, text: str) -> float:
        """Score the whole call so far, not just this window.

        Categories are cumulative across the call: a caller who claims to be the
        CBI at 0:10 and demands an RTGS transfer at 2:00 has done both, even
        though no single 4-second window contains both.
        """
        flat = normalise(text)
        for category in CATEGORIES:
            if category.id in self.matched:
                continue
            phrase = category.match(flat)
            if phrase:
                self.matched[category.id] = phrase

        total = sum(c.weight for c in CATEGORIES if c.id in self.matched)
        return min(100.0, total)

    def analyze(self, pcm_bytes: bytes, timestamp_ms: float) -> Optional[dict]:
        """Transcribe one window and return the updated intent verdict."""
        if not self.available:
            return None

        try:
            text = self.transcribe(pcm_bytes)
        except Exception as exc:
            log.warning("Transcription failed on a window: %s", exc)
            return None

        if not text:
            return None

        previously = set(self.matched)
        self.transcript = (self.transcript + " " + text).strip()[-TRANSCRIPT_LIMIT:]
        score = self.score_text(self.transcript)

        triggers = [
            {
                "id": c.id,
                "label": c.label,
                "detail": f'matched: "{self.matched[c.id]}"',
            }
            for c in CATEGORIES
            if c.id in self.matched and c.id not in previously
        ]

        log.info("[intent] score=%.0f text=%r", score, text[:80])
        return {"score": score, "transcript": self.transcript, "triggers": triggers}
