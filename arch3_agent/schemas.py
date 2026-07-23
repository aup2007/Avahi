from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Single home for all of Arch 3's Pydantic schemas (per-module dataclasses like
# PayoutResult / DamageInstance stay where they are and are reused by import).
# common/ is logic-only and shared with Arch 2, so agent-only schemas live here.


# --- investigation checks (the tool allowlist) ------------------------------


class CheckName(str, Enum):
    """The allowlisted investigation checks. This enum IS the tool allowlist --
    the Planner may only ever choose one of these, and no payout/approval/
    policy-modification action is a member, so the agent cannot select one."""

    IMAGE_QUALITY = "image_quality"
    DAMAGE_TYPE = "damage_type"
    STORY_CONSISTENCY = "story_consistency"
    CLAIMED_PERIL = "claimed_peril"
    POLICY_COVERAGE = "policy_coverage"
    POLICY_DATES = "policy_dates"


# Prerequisite graph: a check is "available" only once its prerequisites are
# resolved (you can't judge story consistency before you know the damage, or
# check coverage before you know the peril). The Planner picks among available
# checks; the Executor enforces the prereqs.
CHECK_PREREQS: dict[CheckName, list[CheckName]] = {
    CheckName.IMAGE_QUALITY: [],
    CheckName.DAMAGE_TYPE: [CheckName.IMAGE_QUALITY],
    CheckName.STORY_CONSISTENCY: [CheckName.DAMAGE_TYPE],
    CheckName.CLAIMED_PERIL: [],
    CheckName.POLICY_COVERAGE: [CheckName.CLAIMED_PERIL],
    CheckName.POLICY_DATES: [],
}

# All six must be resolved before the evidence package is sufficient to hand to
# the deterministic rules engine.
REQUIRED_CHECKS: tuple[CheckName, ...] = tuple(CheckName)


# --- evidence (what the agent produces) -------------------------------------


class EvidenceItem(BaseModel):
    """One resolved investigation result. `status` is the only routing-relevant
    field: 'ok' = clean fact gathered; 'flagged' = a problem the Verifier must
    act on (bad photo, no damage, inconsistency, injection, ...)."""

    check: CheckName
    status: Literal["ok", "flagged"]
    detail: str = ""
    data: dict = Field(default_factory=dict)


class EvidencePackage(BaseModel):
    """Accumulating evidence -- the ONLY thing the agent produces. The
    deterministic rules engine consumes it. The agent never puts a payout,
    approval, or policy change in here."""

    items: dict[CheckName, EvidenceItem] = Field(default_factory=dict)

    def resolved(self) -> set[CheckName]:
        return set(self.items)

    def flagged(self) -> list[EvidenceItem]:
        return [i for i in self.items.values() if i.status == "flagged"]

    def added(self, item: EvidenceItem) -> "EvidencePackage":
        merged = dict(self.items)
        merged[item.check] = item
        return EvidencePackage(items=merged)


# --- LLM outputs (validated at every boundary) ------------------------------


class PlanDecision(BaseModel):
    """Planner LLM output. The LLM's only powers are: declare the evidence
    sufficient, or name the next allowlisted check to investigate."""

    sufficient: bool = False
    next_check: Optional[CheckName] = None
    reasoning: str = ""


class VerifierDecision(BaseModel):
    """Verifier output. Deterministic in the common path; a schema either way so
    an LLM-assisted verifier stays validated and injection-contained."""

    decision: Literal["continue", "complete", "escalate"]
    reasons: list[str] = Field(default_factory=list)


class StoryMatchResult(BaseModel):
    """Semantic-judgment output for the story_consistency check (moved here from
    tools.py so every Arch 3 schema lives in one place)."""

    consistent: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: float) -> float:
        # The model occasionally emits e.g. 1.2 or -0.1; clamp rather than
        # reject, so a formatting slip never crashes an otherwise-valid verdict.
        return min(max(float(v), 0.0), 1.0)


class PerilResult(BaseModel):
    """Semantic-judgment output for the claimed_peril check: which coverage the
    described incident implies (collision vs comprehensive), or unknown."""

    peril: Literal["collision", "comprehensive", "unknown"]
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(max(float(v), 0.0), 1.0)


class TriageAction(BaseModel):
    """Structured triage action (escalate / request re-upload). Records the
    action rather than performing I/O -- no live adjuster queue in this build."""

    action: str
    reasons: list[str] = Field(default_factory=list)
