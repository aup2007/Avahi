import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import rules_engine  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "avahi.db"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_sample_truth.json"

# Truth = "what should happen if perception were perfect", so confidence is
# pinned to 1.0 here. The models' real (imperfect) confidence only matters
# later, when the architectures are scored against this truth.
TRUTH_CONFIDENCE = 1.0

SAMPLE_SIZE = 10


def true_damage(conn: sqlite3.Connection, claim_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT damage_category, severity, coverage_type FROM claim_damage_instances WHERE claim_id = ? ORDER BY id",
        (claim_id,),
    ).fetchall()
    return [{"damage_category": dc, "severity": s, "coverage_type": ct} for dc, s, ct in rows]


def _signature(claim_meta: dict, result) -> tuple:
    # A claim's "kind" for diversity: its route, whether payout is zero, its
    # primary reason, and the coverage type in play -- so the 10 picked span
    # distinct outcomes rather than 10 near-identical approvals.
    payout_zero = (result.payout is not None and result.payout == 0)
    primary_reason = result.reasons[0] if result.reasons else ""
    return (result.route, payout_zero, primary_reason, claim_meta["coverage_type"])


def select_diverse(conn: sqlite3.Connection, n: int) -> list[str]:
    claims = conn.execute(
        """
        SELECT c.claim_id,
               (SELECT coverage_type FROM claim_damage_instances d WHERE d.claim_id = c.claim_id LIMIT 1)
        FROM claims c ORDER BY c.claim_id
        """
    ).fetchall()

    seen_signatures: set = set()
    picked: list[str] = []
    overflow: list[str] = []

    for claim_id, coverage_type in claims:
        result = rules_engine.compute_payout(conn, claim_id, TRUTH_CONFIDENCE)
        sig = _signature({"coverage_type": coverage_type}, result)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            picked.append(claim_id)
        else:
            overflow.append(claim_id)
        if len(picked) >= n:
            break

    # If distinct outcomes were fewer than n, top up with other claims so the
    # sample still has n rows.
    for claim_id in overflow:
        if len(picked) >= n:
            break
        picked.append(claim_id)

    return picked[:n]


def compute(conn: sqlite3.Connection, claim_ids: list[str]) -> list[dict]:
    records = []
    for claim_id in claim_ids:
        result = rules_engine.compute_payout(conn, claim_id, TRUTH_CONFIDENCE)
        row = conn.execute(
            "SELECT customer_id, photo_file, claim_story FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        records.append({
            "claim_id": claim_id,
            "customer_id": row[0],
            "photo_file": row[1],
            "claim_story": row[2],
            "true_damage": true_damage(conn, claim_id),
            "route": result.route,
            "payout": result.payout,
            "deductible_applied": result.deductible_applied,
            "total_cost": result.total_cost,
            "covered_cost": result.covered_cost,
            "reasons": result.reasons,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        claim_ids = select_diverse(conn, args.n)
        records = compute(conn, claim_ids)
    finally:
        conn.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} golden-truth records to {OUT_PATH}")
    for r in records:
        print(f"  {r['claim_id']}  {r['route']:<12} payout={r['payout']} "
              f"deductible={r['deductible_applied']} reasons={r['reasons']}")


if __name__ == "__main__":
    main()
