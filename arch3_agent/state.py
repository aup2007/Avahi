import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field

from arch2_split.damage_assessor import DamageInstance
from arch2_split.gatekeeper import GatekeeperResult
from arch2_split.segmenter import PanelObservation
from common.rules_engine import PayoutResult

from arch3_agent.tools import StoryMatchResult

# The retake loop (gatekeep -> request_better_photo -> gatekeep) is bounded: a
# non-deterministic gatekeeper can recover on a retry, but after this many
# failed attempts the claim goes to a human instead of looping.
MAX_RETAKE_ATTEMPTS = 3

# Overall graph step guard. The flow is short and acyclic apart from the bounded
# retake loop, so this is generous headroom -- it is also the ceiling a future
# multi-step ReAct triage node would be bounded by (SPEC.md §5).
RECURSION_LIMIT = 20


class ClaimInput(BaseModel):
    """Boundary shape/length contract (fast, no I/O). File-existence and
    policy-resolvable checks live in nodes.validate_node (they need the
    filesystem and the DB connection)."""

    claim_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    claim_story: Optional[str] = Field(default=None, max_length=8000)


class ClaimState(TypedDict, total=False):
    """State threaded through the graph. `trajectory` and `reasons` accumulate
    across nodes (operator.add reducer); every other field is written at most
    once per run and last-write-wins."""

    # inputs (write-once)
    conn: Any
    claim_id: str
    image_path: str
    claim_story: Optional[str]
    # working state
    valid: bool
    attempt: int
    gatekeeper: Optional[GatekeeperResult]
    panels: list[PanelObservation]
    damage: list[DamageInstance]
    confidence: float
    story_match: Optional[StoryMatchResult]
    payout_result: Optional[PayoutResult]
    # outputs
    route: str
    payout: Optional[float]
    deductible_applied: Optional[float]
    reasons: Annotated[list[str], operator.add]
    trajectory: Annotated[list[dict], operator.add]


@dataclass
class Arch3Result:
    claim_id: str
    image_path: str
    route: str
    payout: Optional[float]
    deductible_applied: Optional[float]
    confidence: float
    reasons: list[str]
    gatekeeper: Optional[GatekeeperResult]
    panels: list[PanelObservation]
    damage: list[DamageInstance]
    story_match: Optional[StoryMatchResult]
    payout_result: Optional[PayoutResult]
    trajectory: list[dict] = field(default_factory=list)


def traj(tool: str, summary: str) -> dict:
    """One trajectory entry: which tool ran and a one-line summary of its result."""
    return {"tool": tool, "summary": summary}
