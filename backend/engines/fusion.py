"""Risk fusion: turns raw engine output into the score the HUD renders.

    Risk = 0.55 * Physics_Score + 0.45 * Intent_Score

The physics half is derived here from the per-turn events that
`ConversationalPhysicsEngine` emits; the intent half is supplied whole by the
intent engine. If one engine is unavailable (dependency missing, or its owner
has not landed it yet) its weight is redistributed rather than silently scored
as zero, so a partial system still produces an honest number for the demo.
"""

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

PHYSICS_WEIGHT = 0.55
INTENT_WEIGHT = 0.45

# Fluent human turn-taking lands well under 750 ms. A full VAD -> LLM -> TTS
# pipeline cannot. Below FLOOR we score nothing; at CEILING we score 100.
LATENCY_FLOOR_MS = 750.0
LATENCY_CEILING_MS = 2500.0

# One suspicious pause proves little; the pattern is what matters.
TURN_SMOOTHING = 0.45          # EMA weight on the newest turn
CONFIDENCE_TURNS = 3           # turns needed before physics is fully trusted
BARGE_IN_PENALTY = 12.0        # per failure-to-yield
BARGE_IN_CAP = 36.0

ALERT_THRESHOLD = 75.0         # HUD flips to alert / desktop notification fires


@dataclass
class Trigger:
    """A single human-readable reason the score moved."""

    id: str
    label: str
    detail: str
    source: str                # "physics" | "intent"
    ts_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def latency_points(delta_ms: float) -> float:
    """Map one floor-to-floor latency onto 0-100."""
    if delta_ms <= LATENCY_FLOOR_MS:
        return 0.0
    span = LATENCY_CEILING_MS - LATENCY_FLOOR_MS
    return _clamp((delta_ms - LATENCY_FLOOR_MS) / span * 100.0)


class PhysicsScorer:
    """Rolling score over conversational turn dynamics."""

    def __init__(self) -> None:
        self.turns = 0
        self.latencies_ms: deque = deque(maxlen=20)
        self.barge_in_failures = 0
        self._ema: Optional[float] = None

    def record_latency(self, delta_ms: float) -> Optional[Trigger]:
        self.turns += 1
        self.latencies_ms.append(delta_ms)
        points = latency_points(delta_ms)
        self._ema = points if self._ema is None else (
            TURN_SMOOTHING * points + (1 - TURN_SMOOTHING) * self._ema
        )
        if delta_ms >= LATENCY_FLOOR_MS:
            return Trigger(
                id="pipeline_delay",
                label="Pipeline Delay",
                detail=f"{int(round(delta_ms))}ms response gap (human baseline < 750ms)",
                source="physics",
            )
        return None

    def record_barge_in_failure(self) -> Trigger:
        """Remote kept talking through the user, with no human yield reflex."""
        self.barge_in_failures += 1
        return Trigger(
            id="yield_failure",
            label="Yield Failure",
            detail="Caller spoke through interruption without yielding the floor",
            source="physics",
        )

    @property
    def median_latency_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    @property
    def has_evidence(self) -> bool:
        return self.turns > 0 or self.barge_in_failures > 0

    @property
    def score(self) -> float:
        base = self._ema or 0.0
        # Ramp in slowly: a single long pause should not max the dial.
        confidence = min(self.turns / CONFIDENCE_TURNS, 1.0) if self.turns else 0.0
        barge_in = min(self.barge_in_failures * BARGE_IN_PENALTY, BARGE_IN_CAP)
        return _clamp(base * confidence + barge_in)


class FusionScorer:
    """Per-session aggregate of physics + intent, rendered for the HUD."""

    MAX_TRIGGERS = 8

    def __init__(self) -> None:
        self.physics = PhysicsScorer()
        self.intent_score = 0.0
        self.intent_available = False
        self.transcript: str = ""
        self.triggers: deque = deque(maxlen=self.MAX_TRIGGERS)
        self._alerted = False

    # -- ingestion ---------------------------------------------------------
    def update_physics(self, result: dict) -> None:
        """Consume one dict from `ConversationalPhysicsEngine.ingest_pcm`."""
        if not result:
            return
        latency = result.get("latency_ms")
        if latency is not None:
            trigger = self.physics.record_latency(float(latency))
            if trigger:
                self._add_trigger(trigger)
        if result.get("flag") == "YIELD_FAILURE":
            self._add_trigger(self.physics.record_barge_in_failure())

    def update_intent(self, result: dict) -> None:
        """Consume one dict from the intent engine.

        Expected shape:
            {"score": 0-100, "transcript": str,
             "triggers": [{"id", "label", "detail"}, ...]}
        """
        if not result:
            return
        self.intent_available = True
        if result.get("score") is not None:
            self.intent_score = _clamp(float(result["score"]))
        if result.get("transcript"):
            self.transcript = result["transcript"]
        for raw in result.get("triggers", []) or []:
            self._add_trigger(
                Trigger(
                    id=raw.get("id", "intent"),
                    label=raw.get("label", "Coercion Detected"),
                    detail=raw.get("detail", ""),
                    source="intent",
                )
            )

    def _add_trigger(self, trigger: Trigger) -> None:
        # Collapse repeats of the same trigger id onto the newest occurrence.
        for existing in list(self.triggers):
            if existing.id == trigger.id:
                self.triggers.remove(existing)
        self.triggers.append(trigger)

    # -- output ------------------------------------------------------------
    @property
    def risk(self) -> float:
        physics_live = self.physics.has_evidence
        if physics_live and self.intent_available:
            return _clamp(
                PHYSICS_WEIGHT * self.physics.score + INTENT_WEIGHT * self.intent_score
            )
        # Redistribute the missing weight instead of scoring that engine zero.
        if physics_live:
            return _clamp(self.physics.score)
        if self.intent_available:
            return _clamp(self.intent_score)
        return 0.0

    @property
    def band(self) -> str:
        risk = self.risk
        if risk >= ALERT_THRESHOLD:
            return "critical"
        if risk >= 45.0:
            return "elevated"
        return "normal"

    def snapshot(self) -> dict:
        """The payload broadcast to the extension HUD on every update."""
        risk = round(self.risk, 1)
        newly_alerting = risk >= ALERT_THRESHOLD and not self._alerted
        if newly_alerting:
            self._alerted = True
        elif risk < ALERT_THRESHOLD:
            self._alerted = False

        return {
            "type": "risk_update",
            "risk": risk,
            "band": self.band,
            "alert": newly_alerting,
            "physics": {
                "score": round(self.physics.score, 1),
                "available": self.physics.has_evidence,
                "turns": self.physics.turns,
                "last_latency_ms": (
                    round(self.physics.latencies_ms[-1], 1)
                    if self.physics.latencies_ms else None
                ),
                "median_latency_ms": (
                    round(self.physics.median_latency_ms, 1)
                    if self.physics.median_latency_ms is not None else None
                ),
                "yield_failures": self.physics.barge_in_failures,
            },
            "intent": {
                "score": round(self.intent_score, 1),
                "available": self.intent_available,
                "transcript": self.transcript[-280:],
            },
            "weights": {"physics": PHYSICS_WEIGHT, "intent": INTENT_WEIGHT},
            "triggers": [t.to_dict() for t in self.triggers],
            "ts_ms": time.time() * 1000.0,
        }
