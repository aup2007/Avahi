import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from arch2_split import damage_assessor, gatekeeper, segmenter
from arch2_split.damage_assessor import DamageInstance
from arch2_split.gatekeeper import GatekeeperResult
from arch2_split.segmenter import PanelObservation
from common import rules_engine
from common.rules_engine import PayoutResult


@dataclass
class StageLog:
    stage: str
    output: object
    confidence: float | None


@dataclass
class PipelineResult:
    claim_id: str
    image_path: str
    route: str  # "retake" | "auto_approve" | "auto_deny" | "escalate"
    payout: float | None
    deductible_applied: float | None
    confidence: float
    gatekeeper: GatekeeperResult
    panels: list[PanelObservation]
    damage: list[DamageInstance]
    payout_result: PayoutResult | None
    reasons: list[str] = field(default_factory=list)
    stages: list[StageLog] = field(default_factory=list)


def _perception_confidence(gate: GatekeeperResult, damage: list[DamageInstance]) -> float:
    # Weakest-link across damage instances -- one low-confidence severity call
    # should be able to trip escalation on its own (SPEC.md §1). Fall back to
    # the gatekeeper's confidence when the photo shows no assessable damage.
    if damage:
        return min(d.confidence for d in damage)
    return gate.confidence


def run_claim(conn: sqlite3.Connection, claim_id: str, image_path: str) -> PipelineResult:
    stages: list[StageLog] = []

    gate = gatekeeper.check_photo(image_path)
    stages.append(StageLog("gatekeeper", gate, gate.confidence))

    if not gate.valid:
        return PipelineResult(
            claim_id=claim_id, image_path=image_path, route="retake",
            payout=None, deductible_applied=None, confidence=gate.confidence,
            gatekeeper=gate, panels=[], damage=[], payout_result=None,
            reasons=[f"gatekeeper_rejected: {gate.reason}"], stages=stages,
        )

    panels = segmenter.segment_panels(image_path)
    stages.append(StageLog("segmenter", panels, None))  # audit-log only, no gating

    damage = damage_assessor.assess_damage(image_path)
    confidence = _perception_confidence(gate, damage)
    stages.append(StageLog("damage_assessor", damage, confidence))

    # Perception saw nothing while the claim asserts damage -- a disagreement
    # (wrong photo / mismatched claim / perception miss). Never auto-approve
    # through it; hand to a human. Deny paths (lapsed/not-covered) are handled
    # inside compute_payout below and are correct regardless of what vision saw,
    # so this only guards the auto-approve path.
    if not damage:
        return PipelineResult(
            claim_id=claim_id, image_path=image_path, route="escalate",
            payout=None, deductible_applied=None, confidence=confidence,
            gatekeeper=gate, panels=panels, damage=damage, payout_result=None,
            reasons=["no_damage_detected"], stages=stages,
        )

    # Payout uses the DB's authoritative claim_damage_instances (via claim_id),
    # not the vision prediction -- money is never model-dependent. Vision here
    # only gatekeeps and supplies the confidence that gates escalation.
    payout_result = rules_engine.compute_payout(conn, claim_id, confidence)
    stages.append(StageLog("rules_engine", payout_result, None))

    return PipelineResult(
        claim_id=claim_id, image_path=image_path, route=payout_result.route,
        payout=payout_result.payout, deductible_applied=payout_result.deductible_applied,
        confidence=confidence, gatekeeper=gate, panels=panels, damage=damage,
        payout_result=payout_result, reasons=list(payout_result.reasons), stages=stages,
    )


# coverage_type is required by the schema but has no meaning on the live path:
# Option 1 (single pool) never reads it (DEPLOY_PLAN "Coverage handling").
# Stored as a fixed placeholder so the row is schema-valid; the live payout
# sums total cost regardless of this value.
_LIVE_COVERAGE_TYPE = "collision"


def _write_live_claim(
    conn: sqlite3.Connection, customer_id: str, image_path: str, damage: list[DamageInstance]
) -> str:
    claim_id = "live-" + uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO claims (claim_id, customer_id, photo_file, claim_story, claim_date) VALUES (?, ?, ?, ?, ?)",
        (claim_id, customer_id, Path(image_path).name, None, date.today().isoformat()),
    )
    conn.executemany(
        "INSERT INTO claim_damage_instances (claim_id, damage_category, severity, coverage_type) "
        "VALUES (?, ?, ?, ?)",
        [(claim_id, d.damage_category, d.severity, _LIVE_COVERAGE_TYPE) for d in damage],
    )
    conn.commit()
    return claim_id


def run_upload(conn: sqlite3.Connection, customer_id: str, image_path: str) -> PipelineResult:
    # Live-upload intake (DEPLOY_PLAN): the uploaded photo *is* a new claim.
    # Run perception once, write the vision-detected damage as the claim's
    # damage of record, then adjudicate via the Option-1 single-pool branch.
    # run_claim (the eval path) is untouched.
    stages: list[StageLog] = []

    gate = gatekeeper.check_photo(image_path)
    stages.append(StageLog("gatekeeper", gate, gate.confidence))

    if not gate.valid:
        return PipelineResult(
            claim_id="", image_path=image_path, route="retake",
            payout=None, deductible_applied=None, confidence=gate.confidence,
            gatekeeper=gate, panels=[], damage=[], payout_result=None,
            reasons=[f"gatekeeper_rejected: {gate.reason}"], stages=stages,
        )

    panels = segmenter.segment_panels(image_path)
    stages.append(StageLog("segmenter", panels, None))

    damage = damage_assessor.assess_damage(image_path)
    confidence = _perception_confidence(gate, damage)
    stages.append(StageLog("damage_assessor", damage, confidence))

    if not damage:
        return PipelineResult(
            claim_id="", image_path=image_path, route="escalate",
            payout=None, deductible_applied=None, confidence=confidence,
            gatekeeper=gate, panels=panels, damage=damage, payout_result=None,
            reasons=["no_damage_detected"], stages=stages,
        )

    claim_id = _write_live_claim(conn, customer_id, image_path, damage)

    payout_result = rules_engine.compute_payout_live(conn, claim_id, confidence)
    stages.append(StageLog("rules_engine", payout_result, None))

    return PipelineResult(
        claim_id=claim_id, image_path=image_path, route=payout_result.route,
        payout=payout_result.payout, deductible_applied=payout_result.deductible_applied,
        confidence=confidence, gatekeeper=gate, panels=panels, damage=damage,
        payout_result=payout_result, reasons=list(payout_result.reasons), stages=stages,
    )
