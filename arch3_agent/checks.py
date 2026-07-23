import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from arch3_agent import tools
from arch3_agent.schemas import CheckName, EvidenceItem, EvidencePackage


@dataclass
class InvestigationContext:
    """What a check may read. `workspace` carries raw perception objects between
    dependent checks (e.g. damage_type -> story_consistency) without putting
    non-serializable objects into the EvidencePackage the rules engine consumes."""

    conn: sqlite3.Connection
    claim_id: str
    image_path: str
    claim_story: Optional[str]
    workspace: dict = field(default_factory=dict)


# --- the 6 allowlisted investigation checks ---------------------------------
# Perception (image/damage) -> VLM. Semantic (story/peril) -> JUDGE_MODEL.
# Policy (coverage/dates) -> deterministic DB reads, no model at all.
# None of these can approve, deny, price, or modify a policy.


def check_image_quality(ctx: InvestigationContext) -> EvidenceItem:
    gate = tools.gatekeep(ctx.image_path)  # VLM
    return EvidenceItem(
        check=CheckName.IMAGE_QUALITY,
        status="ok" if gate.valid else "flagged",
        detail=gate.reason,
        data={"valid": gate.valid, "confidence": gate.confidence},
    )


def check_damage_type(ctx: InvestigationContext) -> EvidenceItem:
    damage = tools.assess_damage(ctx.image_path)  # VLM
    ctx.workspace["damage"] = damage  # raw objects for story_consistency
    counts = Counter((d.damage_category, d.severity) for d in damage)
    summary = ", ".join(f"{n}x {cat} ({sev})" for (cat, sev), n in counts.items())
    return EvidenceItem(
        check=CheckName.DAMAGE_TYPE,
        status="ok" if damage else "flagged",
        detail=summary or "no damage detected",
        data={
            "instances": [
                {"damage_category": d.damage_category, "severity": d.severity, "confidence": d.confidence}
                for d in damage
            ],
            "confidence": min((d.confidence for d in damage), default=0.0),
        },
    )


def check_story_consistency(ctx: InvestigationContext) -> EvidenceItem:
    damage = ctx.workspace.get("damage", [])  # from check_damage_type (prereq)
    sm = tools.match_damage_to_story(ctx.claim_story, damage)  # JUDGE_MODEL
    if sm.reason == "possible_prompt_injection":
        reason_code = "possible_prompt_injection"
    elif sm.consistent:
        reason_code = "consistent"
    else:
        reason_code = "story_damage_inconsistent"
    return EvidenceItem(
        check=CheckName.STORY_CONSISTENCY,
        status="ok" if sm.consistent else "flagged",
        detail=sm.reason,
        data={"consistent": sm.consistent, "reason_code": reason_code, "confidence": sm.confidence},
    )


def check_claimed_peril(ctx: InvestigationContext) -> EvidenceItem:
    res = tools.interpret_peril(ctx.claim_story)  # JUDGE_MODEL
    ctx.workspace["peril"] = res.peril  # for check_policy_coverage (prereq)
    return EvidenceItem(
        check=CheckName.CLAIMED_PERIL,
        status="ok" if res.peril != "unknown" else "flagged",
        detail=res.reason,
        data={"peril": res.peril, "reason_code": res.reason, "confidence": res.confidence},
    )


def check_policy_coverage(ctx: InvestigationContext) -> EvidenceItem:
    # Deterministic. Records whether the claimed peril is covered -- a FACT for
    # the rules engine, not a decision. "Not covered" is not flagged here (the
    # rules engine deterministically denies it); only an unknown peril flags,
    # since coverage can't be checked without one.
    peril = ctx.workspace.get("peril", "unknown")
    policy = tools.lookup_policy(ctx.conn, ctx.claim_id)
    if policy is None:
        raise ValueError(f"no policy for claim_id={ctx.claim_id!r}")

    covered = {
        "collision": policy["collision_active"],
        "comprehensive": policy["comprehensive_active"],
    }.get(peril)
    return EvidenceItem(
        check=CheckName.POLICY_COVERAGE,
        status="flagged" if peril == "unknown" else "ok",
        detail=f"peril={peril} covered={covered}",
        data={
            "peril": peril,
            "collision_active": policy["collision_active"],
            "comprehensive_active": policy["comprehensive_active"],
            "covered_for_peril": covered,
        },
    )


def check_policy_dates(ctx: InvestigationContext) -> EvidenceItem:
    # Deterministic. Records active/lapsed -- a FACT. A lapsed policy is denied
    # by the rules engine at adjudication, never by the agent.
    policy = tools.lookup_policy(ctx.conn, ctx.claim_id)
    if policy is None:
        raise ValueError(f"no policy for claim_id={ctx.claim_id!r}")
    return EvidenceItem(
        check=CheckName.POLICY_DATES,
        status="ok",
        detail=policy["policy_status"],
        data={"policy_status": policy["policy_status"]},
    )


# THE ALLOWLIST. The Executor may only dispatch a check found in this registry;
# there is deliberately no payout/approve/deny/policy-write function here, so the
# agent cannot invoke one no matter what the Planner proposes.
CHECK_REGISTRY: dict[CheckName, Callable[[InvestigationContext], EvidenceItem]] = {
    CheckName.IMAGE_QUALITY: check_image_quality,
    CheckName.DAMAGE_TYPE: check_damage_type,
    CheckName.STORY_CONSISTENCY: check_story_consistency,
    CheckName.CLAIMED_PERIL: check_claimed_peril,
    CheckName.POLICY_COVERAGE: check_policy_coverage,
    CheckName.POLICY_DATES: check_policy_dates,
}


def perception_confidence(evidence: EvidencePackage) -> float:
    # Weakest-link damage confidence (mirrors Arch 2 SPEC.md §1), falling back to
    # the gatekeeper's confidence, then 0.0. Passed to the rules engine so a
    # low-confidence perception can still gate escalation at adjudication.
    dmg = evidence.items.get(CheckName.DAMAGE_TYPE)
    if dmg and dmg.data.get("instances"):
        return min(i["confidence"] for i in dmg.data["instances"])
    img = evidence.items.get(CheckName.IMAGE_QUALITY)
    if img:
        return float(img.data.get("confidence", 0.0))
    return 0.0
