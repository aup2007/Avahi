import argparse
import csv
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "golden_sample_truth.json"
MISMATCH_SRC = REPO / "data" / "mismatch_truth.json"
OUT_DIR = REPO / "data" / "golden_set"

VERSION = "v2"


def export(version: str) -> None:
    records = json.loads(SRC.read_text())
    # v2 adds the labelled story-vs-damage mismatch subset (escalate/
    # story_damage_inconsistent) so the eval can score the dimension where the
    # architectures most differ. Base records carry mismatch_type=None.
    for r in records:
        r.setdefault("mismatch_type", None)
    if MISMATCH_SRC.exists():
        records = records + json.loads(MISMATCH_SRC.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": version,
        "frozen_at": date.today().isoformat(),
        "source": "computed by common/rules_engine over db/avahi.db claims (Plan.md step 7), plus a labelled story-damage mismatch subset (db/seed_mismatch_claims.py)",
        "count": len(records),
        "mismatch_count": sum(1 for r in records if r.get("mismatch_type")),
        "truth_confidence": 1.0,  # truth = outcome under perfect perception
        "note": "Frozen evaluation answer key. Immutable, versioned in git, never queried live, never trained on.",
        "records": records,
    }

    json_path = OUT_DIR / f"golden_set_{version}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = OUT_DIR / f"golden_set_{version}.csv"
    fields = ["claim_id", "customer_id", "photo_file", "claim_story", "mismatch_type", "true_damage",
              "route", "payout", "deductible_applied", "total_cost", "covered_cost", "reasons"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                **{k: r.get(k) for k in fields if k not in ("true_damage", "reasons")},
                "true_damage": json.dumps(r["true_damage"]),
                "reasons": ";".join(r["reasons"]),
            })

    print(f"froze {len(records)} records -> {json_path.relative_to(REPO)} and {csv_path.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()
    export(args.version)
