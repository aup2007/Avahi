import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

# Every live adjudication is recorded here, not just escalations -- the queue is a
# filtered view of this table. The machine's verdict columns (route, payout, ...)
# are written once and never updated; a human reviewer's decision lands in the
# separate review_* columns, so the two are always distinguishable after the fact.
SCHEMA = """
CREATE TABLE IF NOT EXISTS adjudications (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id           TEXT    NOT NULL REFERENCES claims (claim_id),
    arch               TEXT    NOT NULL CHECK (arch IN ('2', '3')),
    route              TEXT    NOT NULL,
    payout             REAL,
    deductible_applied REAL,
    confidence         REAL,
    reasons            TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    evidence           TEXT,                            -- JSON, arch 3 only
    trajectory         TEXT,                            -- JSON, arch 3 only
    created_at         TEXT    NOT NULL,
    review_status      TEXT    NOT NULL DEFAULT 'pending'
                               CHECK (review_status IN ('pending', 'resolved', 'not_required')),
    review_decision    TEXT    CHECK (review_decision IN ('approved', 'denied', 'need_info')),
    review_note        TEXT,
    reviewed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_adjudications_queue
    ON adjudications (review_status, created_at);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(conn: sqlite3.Connection, arch: str, result: dict) -> Optional[int]:
    """Log one adjudication. Only escalations enter the review queue; everything
    else is logged as not_required so the audit trail stays complete."""
    claim_id = result.get("claim_id")
    if not claim_id:
        return None  # rejected before a claim existed (e.g. gatekeeper retake)

    route = result.get("route", "")
    cur = conn.execute(
        "INSERT INTO adjudications (claim_id, arch, route, payout, deductible_applied, "
        "confidence, reasons, evidence, trajectory, created_at, review_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            claim_id, arch, route, result.get("payout"), result.get("deductible_applied"),
            result.get("confidence"), json.dumps(result.get("reasons") or []),
            json.dumps(result["evidence"]) if result.get("evidence") else None,
            json.dumps(result["trajectory"]) if result.get("trajectory") else None,
            _now(), "pending" if route == "escalate" else "not_required",
        ),
    )
    conn.commit()
    return cur.lastrowid


def pending_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM adjudications WHERE review_status = 'pending'"
    ).fetchone()[0]


def queue(conn: sqlite3.Connection, status: str = "pending", limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT a.*, c.customer_id, c.claim_story, c.photo_file, p.policy_data "
        "FROM adjudications a "
        "JOIN claims c ON c.claim_id = a.claim_id "
        "LEFT JOIN policies p ON p.customer_id = c.customer_id "
        "WHERE a.review_status = ? ORDER BY a.created_at DESC LIMIT ?",
        (status, limit),
    ).fetchall()

    out = []
    for r in rows:
        policy = json.loads(r["policy_data"]) if r["policy_data"] else {}
        out.append({
            "id": r["id"],
            "claim_id": r["claim_id"],
            "customer_id": r["customer_id"],
            "customer_name": policy.get("name"),
            "claim_story": r["claim_story"],
            "arch": r["arch"],
            "route": r["route"],
            "payout": r["payout"],
            "confidence": r["confidence"],
            "reasons": json.loads(r["reasons"]),
            "evidence": json.loads(r["evidence"]) if r["evidence"] else None,
            "trajectory": json.loads(r["trajectory"]) if r["trajectory"] else None,
            "created_at": r["created_at"],
            "review_status": r["review_status"],
            "review_decision": r["review_decision"],
            "review_note": r["review_note"],
            "reviewed_at": r["reviewed_at"],
        })
    return out


def resolve(conn: sqlite3.Connection, adjudication_id: int, decision: str, note: str) -> bool:
    # Deliberately does NOT touch route/payout: the rules engine's verdict is the
    # permanent record of what the system decided, and the human's call sits beside
    # it. An admin overturning a denial is visible as exactly that.
    cur = conn.execute(
        "UPDATE adjudications SET review_status = 'resolved', review_decision = ?, "
        "review_note = ?, reviewed_at = ? WHERE id = ? AND review_status = 'pending'",
        (decision, note or None, _now(), adjudication_id),
    )
    conn.commit()
    return cur.rowcount > 0
