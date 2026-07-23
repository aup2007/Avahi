from common import rules_engine

from arch3_agent.checks import CHECK_REGISTRY, InvestigationContext, perception_confidence
from arch3_agent.config import PLANNER_MODEL
from arch3_agent.live_claim import persist_live_damage
from arch3_agent.llm import call_text_json
from arch3_agent.pev_state import MAX_REPLANS, MAX_TOOL_CALLS, PEVState, traj
from arch3_agent.schemas import (
    CHECK_PREREQS,
    REQUIRED_CHECKS,
    CheckName,
    EvidenceItem,
    EvidencePackage,
    PlanDecision,
)


# --- flag classification (deterministic) ------------------------------------
# Returns (reason_code, is_hard). Hard flags escalate immediately; soft flags
# (ambiguous, not fatal) are eligible for one adaptive re-investigation first.


def classify_flag(item: EvidenceItem) -> tuple[str, bool]:
    data = item.data
    if data.get("error"):
        return "tool_failure", True
    if item.check == CheckName.STORY_CONSISTENCY:
        return data.get("reason_code", "story_damage_inconsistent"), True
    if item.check == CheckName.IMAGE_QUALITY:
        return "image_unusable", True
    if item.check == CheckName.DAMAGE_TYPE:
        return "no_damage_detected", True
    if item.check == CheckName.CLAIMED_PERIL:
        if data.get("reason_code") == "possible_prompt_injection":
            return "possible_prompt_injection", True
        return "peril_unknown", False
    if item.check == CheckName.POLICY_COVERAGE:
        return "coverage_unknown", False
    return "flagged", True


def _prereqs_met(check: CheckName, resolved: set[CheckName]) -> bool:
    return all(p in resolved for p in CHECK_PREREQS[check])


def _all_required_ok(ev: EvidencePackage) -> bool:
    return all(c in ev.resolved() for c in REQUIRED_CHECKS) and not ev.flagged()


# --- Planner (LLM: adaptive tool selection) ---------------------------------

_PLANNER_SYSTEM = (
    "You are the planner for a car-insurance claim investigation agent. Given "
    "the evidence gathered so far and a list of remaining ALLOWED checks, choose "
    "the single most useful next check. You may ONLY choose from the provided "
    "candidates -- never invent a check, never propose paying, approving, "
    "denying, or modifying anything. "
    'Respond strict JSON: {"sufficient": bool, "next_check": <one candidate>, '
    '"reasoning": string}.'
)


def _llm_pick_next(state: PEVState, candidates: list[CheckName]) -> CheckName:
    # Deterministic when there's no real choice; otherwise the small PLANNER_MODEL
    # picks, validated against the candidate allowlist with a deterministic
    # fallback so a bad/failed LLM response can never derail the plan.
    if len(candidates) == 1:
        return candidates[0]
    resolved = sorted(c.value for c in (state.get("evidence") or EvidencePackage()).resolved())
    user = (
        f"Evidence gathered: {resolved or 'none'}\n"
        f"Remaining allowed checks: {[c.value for c in candidates]}\n"
        "Which check should run next?"
    )
    try:
        decision = PlanDecision.model_validate(call_text_json(_PLANNER_SYSTEM, user, model=PLANNER_MODEL))
        if decision.next_check in candidates:
            return decision.next_check
    except Exception:
        pass
    return candidates[0]


def planner_node(state: PEVState) -> dict:
    ev = state.get("evidence") or EvidencePackage()
    resolved = ev.resolved()

    available = [c for c in REQUIRED_CHECKS if c not in resolved and _prereqs_met(c, resolved)]
    if available:
        nxt = _llm_pick_next(state, available)
        return {"plan_next": nxt, "sufficient": False,
                "trajectory": [traj("planner", f"next={nxt.value}")]}

    # All required checks have run. A soft-flagged result may warrant one adaptive
    # re-investigation (a replan) if budget remains.
    retryable = [i.check for i in ev.flagged() if not classify_flag(i)[1]]
    if retryable and state.get("replans", 0) < MAX_REPLANS:
        nxt = _llm_pick_next(state, retryable)
        return {"plan_next": nxt, "sufficient": False,
                "trajectory": [traj("planner", f"replan retry={nxt.value}")]}

    return {"plan_next": None, "sufficient": True,
            "trajectory": [traj("planner", "evidence sufficient")]}


# --- Executor (deterministic: calls exactly one allowlisted tool) -----------


def executor_node(state: PEVState) -> dict:
    check: CheckName = state["plan_next"]
    ev = state.get("evidence") or EvidencePackage()
    workspace = dict(state.get("workspace") or {})
    is_rerun = check in ev.resolved()
    prev = ev.items.get(check)

    ctx = InvestigationContext(
        conn=state["conn"], claim_id=state["claim_id"], image_path=state["image_path"],
        claim_story=state.get("claim_story"), workspace=workspace,
    )
    try:
        item = CHECK_REGISTRY[check](ctx)
    except Exception as e:  # tool failure -> flagged -> Verifier escalates
        item = EvidenceItem(check=check, status="flagged", detail=f"tool_failure: {e}", data={"error": str(e)})

    tool_calls = state.get("tool_calls", 0) + 1
    replans = state.get("replans", 0) + (1 if is_rerun else 0)
    unproductive = bool(is_rerun and prev is not None and prev.status == item.status)
    tag = " rerun" if is_rerun else ""
    return {
        "evidence": ev.added(item), "workspace": workspace,
        "tool_calls": tool_calls, "replans": replans, "unproductive": unproductive,
        "trajectory": [traj("executor", f"{check.value} -> {item.status} (call {tool_calls}{tag})")],
    }


# --- Verifier (deterministic: continue | complete | escalate) ---------------


def _escalate(reasons: list[str]) -> dict:
    return {"decision": "escalate", "reasons": reasons,
            "trajectory": [traj("verifier", "escalate: " + ",".join(reasons))]}


def verifier_node(state: PEVState) -> dict:
    ev = state["evidence"]
    last = state.get("plan_next")
    item = ev.items.get(last) if last else None

    if item is not None and item.status == "flagged":
        code, hard = classify_flag(item)
        if hard:
            return _escalate([code])
        if state.get("replans", 0) >= MAX_REPLANS:  # soft flag, no retry budget
            return _escalate(["missing_evidence", code])

    if state.get("unproductive"):
        return _escalate(["unproductive_repeat"])
    if state.get("replans", 0) > MAX_REPLANS:
        return _escalate(["replan_limit"])
    if _all_required_ok(ev):
        return {"decision": "complete", "trajectory": [traj("verifier", "complete")]}
    if state.get("tool_calls", 0) >= MAX_TOOL_CALLS:
        return _escalate(["tool_call_limit"])
    return {"decision": "continue", "trajectory": [traj("verifier", "continue")]}


# --- deterministic terminals (money + handoff, OUTSIDE the agent loop) -------


def adjudicate_node(state: PEVState) -> dict:
    # The ONLY place money is computed. Runs only on a clean, sufficient evidence
    # package; the rules engine (not the agent) decides approve/deny/escalate.
    confidence = perception_confidence(state["evidence"])
    if state.get("live_customer_id"):
        # Live upload: the photo IS the claim, so the damage the agent perceived
        # has to become rows before the rules engine can price it. Golden-set runs
        # have no live_customer_id and skip this entirely -- the eval path is
        # unchanged, and this still writes only facts, never a dollar figure.
        persist_live_damage(state["conn"], state["claim_id"], state["evidence"])
    pr = rules_engine.compute_payout(state["conn"], state["claim_id"], confidence)
    return {
        "route": pr.route, "payout": pr.payout, "deductible_applied": pr.deductible_applied,
        "confidence": confidence, "reasons": list(pr.reasons),
        "trajectory": [traj("adjudicate", f"route={pr.route} payout={pr.payout}")],
    }


def escalate_terminal_node(state: PEVState) -> dict:
    confidence = perception_confidence(state.get("evidence") or EvidencePackage())
    return {
        "route": "escalate", "payout": None, "deductible_applied": None, "confidence": confidence,
        "trajectory": [traj("escalate", "handed to human with evidence package")],
    }


# --- routing predicates -----------------------------------------------------


def route_after_plan(state: PEVState) -> str:
    return "adjudicate" if state.get("sufficient") else "executor"


def route_after_verify(state: PEVState) -> str:
    return {"continue": "planner", "complete": "adjudicate", "escalate": "escalate"}[state["decision"]]
