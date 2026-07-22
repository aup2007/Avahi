"""The only place payout math happens (CLAUDE.md standing decision).

Plan.md step 10. Coverage check + min(cost, limit) - deductible + the
$2,000/confidence escalation gate. Pure code, no model calls -- this must be
~100% correct against hand-computed cases (unit-tested in
common/test_rules_engine.py), since any miss here is a code bug, not a
model error (SPEC.md §4).

Used by arch2_split/pipeline.py directly and by arch3_agent/tools.py's
compute_payout() wrapper -- never reimplemented, only imported.
"""
import sqlite3
from dataclasses import dataclass, field

from common import cost_lookup, policy_lookup

AUTO_APPROVE_MAX_PAYOUT = 2000.0
# "above the chosen bar" (SPEC.md §4) -- not pinned by SPEC.md to an exact
# number; 0.75 is this build's chosen bar, tunable during Phase 2 eval.
CONFIDENCE_THRESHOLD = 0.75


@dataclass
class PayoutResult:
    route: str  # "auto_approve" | "auto_deny" | "escalate"
    payout: float | None
    deductible_applied: float | None
    total_cost: float
    covered_cost: float  # sum of covered instances' cost, pre-limit-cap, pre-deductible
    reasons: list[str] = field(default_factory=list)


def compute_payout(
    conn: sqlite3.Connection,
    claim_id: str,
    confidence: float,
    *,
    auto_approve_max: float = AUTO_APPROVE_MAX_PAYOUT,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PayoutResult:
    policy = policy_lookup.get_policy_for_claim(conn, claim_id)
    if policy is None:
        raise ValueError(f"no policy found for claim_id={claim_id!r}")

    if policy["policy_status"] == "lapsed":
        return PayoutResult(
            route="auto_deny", payout=None, deductible_applied=None,
            total_cost=0.0, covered_cost=0.0, reasons=["policy_lapsed"],
        )

    costs = cost_lookup.claim_costs_by_coverage(conn, claim_id, policy["car_class"])

    # Each coverage type is its own limit pool (SPEC.md §8, Table 2) -- an
    # instance's cost only counts if that coverage is active, and each
    # active pool is capped by its own limit before summing.
    covered_collision = policy["collision_active"]
    covered_comprehensive = policy["comprehensive_active"]

    capped_collision = min(costs["collision_cost"], policy["collision_limit"]) if covered_collision else 0.0
    capped_comprehensive = (
        min(costs["comprehensive_cost"], policy["comprehensive_limit"]) if covered_comprehensive else 0.0
    )
    covered_cost = capped_collision + capped_comprehensive

    uncovered_present = (
        (costs["collision_cost"] > 0 and not covered_collision)
        or (costs["comprehensive_cost"] > 0 and not covered_comprehensive)
    )

    if covered_cost <= 0:
        return PayoutResult(
            route="auto_deny", payout=None, deductible_applied=None,
            total_cost=costs["total_cost"], covered_cost=0.0, reasons=["not_covered"],
        )

    # Deductible applied once at the policy level, not per-instance (SPEC.md §4).
    payout = max(covered_cost - policy["deductible"], 0.0)

    reasons = []
    if uncovered_present:
        reasons.append("partial_coverage_some_instances_excluded")

    if payout <= auto_approve_max and confidence >= confidence_threshold:
        route = "auto_approve"
    else:
        route = "escalate"
        if payout > auto_approve_max:
            reasons.append("payout_above_auto_approve_threshold")
        if confidence < confidence_threshold:
            reasons.append("confidence_below_threshold")

    return PayoutResult(
        route=route, payout=payout, deductible_applied=policy["deductible"],
        total_cost=costs["total_cost"], covered_cost=covered_cost, reasons=reasons,
    )
