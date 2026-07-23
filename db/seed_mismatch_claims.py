import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import rules_engine  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "avahi.db"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "mismatch_truth.json"

# Labelled story-vs-damage mismatch subset. Each claim reuses a real photo and
# its real DB damage, but pairs it with a story that deliberately contradicts
# what the photo shows. The correct action is human review, so the golden route
# is escalate/story_damage_inconsistent -- authored here, NOT computed by
# rules_engine (which reads damage only and would route these on payout math).
# Every policy is active + fully covered so the escalation is driven purely by
# the inconsistency, not a lapse or coverage deny.
MISMATCH_CLAIMS = [
    {
        "claim_id": "CLM-90001",
        "customer_id": "demo-cust13",
        "photo_file": "001372.jpg",
        "claim_story": "Head-on collision totaled the entire front of the car -- hood, both fenders, and the bumper all need full replacement.",
        "mismatch_type": "story_inflates_severity",
        "damage": [("dent", "minor", "collision")],
    },
    {
        "claim_id": "CLM-90002",
        "customer_id": "demo-cust19",
        "photo_file": "000012.jpg",
        "claim_story": "A rock kicked up on the highway cracked and completely shattered my windshield.",
        "mismatch_type": "fabricated_damage_type",
        "damage": [("tire flat", "moderate", "collision"), ("tire flat", "moderate", "collision")],
    },
    {
        "claim_id": "CLM-90003",
        "customer_id": "demo-cust7",
        "photo_file": "000042.jpg",
        "claim_story": "Minor shopping-cart ding on the rear door, nothing else.",
        "mismatch_type": "story_understates_severity",
        "damage": [
            ("dent", "minor", "collision"), ("dent", "minor", "collision"),
            ("scratch", "minor", "collision"), ("scratch", "minor", "collision"),
            ("crack", "moderate", "collision"), ("crack", "moderate", "collision"),
            ("glass shatter", "severe", "collision"), ("lamp broken", "severe", "collision"),
        ],
    },
    {
        "claim_id": "CLM-90004",
        "customer_id": "demo-cust1",
        "photo_file": "000570.jpg",
        "claim_story": "Rear-ended at a stoplight by another vehicle that failed to brake.",
        "mismatch_type": "peril_mismatch_collision_vs_comprehensive",
        "damage": [
            ("dent", "minor", "comprehensive"), ("dent", "minor", "comprehensive"),
            ("crack", "moderate", "comprehensive"),
        ],
    },
]

ESCALATE_REASON = "story_damage_inconsistent"


def seed(conn: sqlite3.Connection) -> None:
    for claim in MISMATCH_CLAIMS:
        conn.execute("DELETE FROM claim_damage_instances WHERE claim_id = ?", (claim["claim_id"],))
        conn.execute("DELETE FROM claims WHERE claim_id = ?", (claim["claim_id"],))
        conn.execute(
            "INSERT INTO claims (claim_id, customer_id, photo_file, claim_story, claim_date) VALUES (?, ?, ?, ?, ?)",
            (claim["claim_id"], claim["customer_id"], claim["photo_file"], claim["claim_story"], "2026-07-22"),
        )
        conn.executemany(
            "INSERT INTO claim_damage_instances (claim_id, damage_category, severity, coverage_type) VALUES (?, ?, ?, ?)",
            [(claim["claim_id"], dc, sev, cov) for dc, sev, cov in claim["damage"]],
        )
    conn.commit()


def golden_records(conn: sqlite3.Connection) -> list[dict]:
    records = []
    for claim in MISMATCH_CLAIMS:
        # Damage-based cost for reference only; route/payout are overridden to the
        # authored mismatch label so the escalation reflects the inconsistency.
        engine = rules_engine.compute_payout(conn, claim["claim_id"], confidence=1.0)
        records.append({
            "claim_id": claim["claim_id"],
            "customer_id": claim["customer_id"],
            "photo_file": claim["photo_file"],
            "claim_story": claim["claim_story"],
            "mismatch_type": claim["mismatch_type"],
            "true_damage": [
                {"damage_category": dc, "severity": sev, "coverage_type": cov}
                for dc, sev, cov in claim["damage"]
            ],
            "route": "escalate",
            "payout": None,
            "deductible_applied": None,
            "total_cost": engine.total_cost,
            "covered_cost": engine.covered_cost,
            "reasons": [ESCALATE_REASON],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default=str(DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        seed(conn)
        records = golden_records(conn)
    finally:
        conn.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"seeded {len(records)} mismatch claims and wrote truth to {OUT_PATH}")
    for r in records:
        print(f"  {r['claim_id']}  {r['mismatch_type']:<42} "
              f"damage_cost={r['covered_cost']} -> route={r['route']} ({r['reasons'][0]})")


if __name__ == "__main__":
    main()
