import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from arch3_agent.schemas import CheckName, EvidencePackage

# Bounded autonomy. The two limits are deliberately consistent: 6 required checks
# gathered once = 6 tool calls, leaving room for up to 2 adaptive re-investigations
# before the hard ceiling of 8.
MAX_TOOL_CALLS = 8
MAX_REPLANS = 2


class PEVState(TypedDict, total=False):
    """State threaded through the Planner-Executor-Verifier loop. `reasons` and
    `trajectory` accumulate (operator.add); everything else is last-write-wins.
    `workspace` holds raw perception objects for dependent checks and is never
    serialized into the result."""

    # the uploaded claim packet (inputs)
    conn: Any
    claim_id: str
    image_path: str
    claim_story: Optional[str]
    live_customer_id: Optional[str]  # set only on the live-upload path; see intake.py
    # agent working memory
    evidence: EvidencePackage
    workspace: dict
    plan_next: Optional[CheckName]   # Planner's chosen next check
    sufficient: bool                 # Planner: evidence complete?
    tool_calls: int
    replans: int
    unproductive: bool               # last Executor run re-ran a check with no new info
    decision: str                    # Verifier: continue | complete | escalate
    # outputs (set only at the deterministic adjudicate/escalate terminals)
    route: str
    payout: Optional[float]
    deductible_applied: Optional[float]
    confidence: float
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
    evidence: dict          # EvidencePackage.model_dump() -- serializable
    tool_calls: int
    replans: int
    trajectory: list[dict] = field(default_factory=list)


def traj(node: str, summary: str) -> dict:
    return {"node": node, "summary": summary}
