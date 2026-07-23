from pathlib import Path

from arch2_split.damage_assessor import DamageInstance
from arch2_split.gatekeeper import GatekeeperResult

from arch3_agent import tools
from arch3_agent.state import MAX_RETAKE_ATTEMPTS, ClaimState, traj


def _perception_confidence(gate: GatekeeperResult, damage: list[DamageInstance]) -> float:
    # Weakest-link across damage instances, gatekeeper confidence as fallback --
    # mirrors Arch 2's rule (SPEC.md §1). Inlined rather than importing Arch 2's
    # private helper, to keep arch3_agent isolated from arch2 internals.
    if damage:
        return min(d.confidence for d in damage)
    return gate.confidence


# --- nodes (each calls exactly one tool from tools.py) ----------------------


def validate_node(state: ClaimState) -> dict:
    # Deeper half of the input contract: the file must exist and the claim must
    # resolve to a policy. Anything malformed escalates cleanly -- it never
    # crashes mid-graph and never reaches the money.
    errors: list[str] = []
    image_path = state.get("image_path", "")
    if not image_path or not Path(image_path).exists():
        errors.append("image_not_found")
    if tools.lookup_policy(state["conn"], state["claim_id"]) is None:
        errors.append("policy_not_found")

    if errors:
        return {"valid": False, "reasons": errors,
                "trajectory": [traj("validate_input", ",".join(errors))]}
    return {"valid": True, "attempt": 0,
            "trajectory": [traj("validate_input", "ok")]}


def gatekeep_node(state: ClaimState) -> dict:
    attempt = state.get("attempt", 0) + 1
    gate = tools.gatekeep(state["image_path"])
    return {
        "attempt": attempt, "gatekeeper": gate, "confidence": gate.confidence,
        "trajectory": [traj("gatekeep", f"valid={gate.valid} conf={gate.confidence:.2f} attempt={attempt}")],
    }


def retake_node(state: ClaimState) -> dict:
    # Asks for a re-upload, then loops back to gatekeep (edge wired in agent.py).
    action = tools.request_better_photo(f"gatekeeper_rejected: {state['gatekeeper'].reason}")
    return {"trajectory": [traj("request_better_photo", action.reasons[0])]}


def retake_exhausted_node(state: ClaimState) -> dict:
    return {"reasons": ["photo_unusable_after_3_attempts"],
            "trajectory": [traj("retake_exhausted", f"{MAX_RETAKE_ATTEMPTS} attempts failed")]}


def segment_node(state: ClaimState) -> dict:
    panels = tools.segment(state["image_path"])
    return {"panels": panels,
            "trajectory": [traj("segment", f"{len(panels)} panels (audit-only)")]}


def assess_node(state: ClaimState) -> dict:
    damage = tools.assess_damage(state["image_path"])
    confidence = _perception_confidence(state["gatekeeper"], damage)
    out: dict = {
        "damage": damage, "confidence": confidence,
        "trajectory": [traj("assess_damage", f"{len(damage)} instances conf={confidence:.2f}")],
    }
    if not damage:
        out["reasons"] = ["no_damage_detected"]
    return out


def triage_node(state: ClaimState) -> dict:
    # The one genuinely open-ended judgment: does the claimant's story match the
    # detected damage? The LLM here can only ever push toward escalate -- it
    # never approves, denies, or touches the payout (OWASP LLM01 L3 containment).
    sm = tools.match_damage_to_story(state.get("claim_story"), state["damage"])
    out: dict = {"story_match": sm,
                 "trajectory": [traj("match_damage_to_story", f"consistent={sm.consistent} ({sm.reason})")]}
    if not sm.consistent:
        # Canonical reason code for routing / golden comparison; sm.reason (the
        # model's free-text rationale) is preserved in the trajectory above.
        reason = "possible_prompt_injection" if sm.reason == "possible_prompt_injection" else "story_damage_inconsistent"
        out["reasons"] = [reason]
    return out


def adjudicate_node(state: ClaimState) -> dict:
    # Deterministic money. compute_payout reads the DB's claim_damage_instances
    # (not the vision output) and returns the route via the $2k/confidence gate.
    pr = tools.compute_payout(state["conn"], state["claim_id"], state["confidence"])
    return {
        "payout_result": pr, "route": pr.route, "payout": pr.payout,
        "deductible_applied": pr.deductible_applied, "reasons": list(pr.reasons),
        "trajectory": [traj("compute_payout", f"route={pr.route} payout={pr.payout}")],
    }


def escalate_node(state: ClaimState) -> dict:
    # Single terminal escalate route. The accumulated reasons are the split key
    # for the future siu_review / adjuster_review typing (see tools.escalate_to_human).
    action = tools.escalate_to_human(state.get("reasons", []))
    return {"route": "escalate", "payout": None, "deductible_applied": None,
            "trajectory": [traj("escalate_to_human", ",".join(action.reasons))]}


# --- edge-routing predicates (the deterministic backbone's branch logic) ----


def route_after_validate(state: ClaimState) -> str:
    return "gatekeep" if state.get("valid") else "escalate"


def route_after_gate(state: ClaimState) -> str:
    if state["gatekeeper"].valid:
        return "segment"
    if state["attempt"] >= MAX_RETAKE_ATTEMPTS:
        return "exhausted"
    return "retake"


def route_after_assess(state: ClaimState) -> str:
    return "triage" if state.get("damage") else "escalate"


def route_after_triage(state: ClaimState) -> str:
    return "adjudicate" if state["story_match"].consistent else "escalate"
