import sqlite3
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from arch3_agent.schemas import CheckName, EvidencePackage

# A peril the agent could not name can't select a coverage pool. Fall back to
# collision (what Arch 2 assumes for every live upload) rather than guessing
# comprehensive -- but note the agent flags peril_unknown, so a claim that lands
# here is already headed for escalation on its own merits.
_FALLBACK_COVERAGE_TYPE = "collision"


def create_live_claim(
    conn: sqlite3.Connection,
    customer_id: str,
    image_path: str,
    claim_story: Optional[str],
) -> str:
    # The claims row must exist *before* the agent runs: check_policy_coverage and
    # check_policy_dates resolve the policy by joining through claims.claim_id.
    # Damage rows come later, from what the agent perceives (persist_live_damage).
    claim_id = "live3-" + uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO claims (claim_id, customer_id, photo_file, claim_story, claim_date) "
        "VALUES (?, ?, ?, ?, ?)",
        (claim_id, customer_id, Path(image_path).name, claim_story or None, date.today().isoformat()),
    )
    conn.commit()
    return claim_id


def persist_live_damage(
    conn: sqlite3.Connection, claim_id: str, evidence: EvidencePackage
) -> int:
    """Turn the agent's perceived damage into claim_damage_instances rows.

    Facts only -- category and severity come from the vision check, coverage_type
    from the peril the agent read out of the story. No cost, no limit, no payout:
    the rules engine still owns all of that.
    """
    dmg = evidence.items.get(CheckName.DAMAGE_TYPE)
    instances = (dmg.data.get("instances") or []) if dmg else []
    if not instances:
        return 0

    peril_item = evidence.items.get(CheckName.CLAIMED_PERIL)
    peril = (peril_item.data.get("peril") if peril_item else None) or "unknown"
    coverage_type = peril if peril in ("collision", "comprehensive") else _FALLBACK_COVERAGE_TYPE

    # Idempotent: adjudicate runs once per claim, but a retry must not double-bill.
    conn.execute("DELETE FROM claim_damage_instances WHERE claim_id = ?", (claim_id,))
    conn.executemany(
        "INSERT INTO claim_damage_instances (claim_id, damage_category, severity, coverage_type) "
        "VALUES (?, ?, ?, ?)",
        [(claim_id, i["damage_category"], i["severity"], coverage_type) for i in instances],
    )
    conn.commit()
    return len(instances)
