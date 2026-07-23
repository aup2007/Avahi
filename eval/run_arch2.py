import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()

from arch2_split import pipeline  # noqa: E402

DB_PATH = REPO / "db" / "avahi.db"
IMAGE_DIR = REPO / "CarDD_release" / "CarDD_COCO" / "test2017"
GOLDEN_DEFAULT = REPO / "data" / "golden_set" / "golden_set_v2.json"

PAYOUT_TOL = 0.01


def _payout_matches(pred, truth) -> bool:
    if pred is None or truth is None:
        return pred is None and truth is None
    return abs(pred - truth) <= PAYOUT_TOL


def evaluate(golden_path: Path, sleep: float) -> dict:
    payload = json.loads(golden_path.read_text())
    records = payload["records"]
    conn = sqlite3.connect(DB_PATH)

    rows = []
    try:
        for rec in records:
            image_path = str(IMAGE_DIR / rec["photo_file"])
            result = pipeline.run_claim(conn, rec["claim_id"], image_path)
            route_ok = result.route == rec["route"]
            payout_ok = _payout_matches(result.payout, rec["payout"])
            rows.append({
                "claim_id": rec["claim_id"],
                "mismatch_type": rec.get("mismatch_type"),
                "true_route": rec["route"],
                "pred_route": result.route,
                "route_ok": route_ok,
                "true_payout": rec["payout"],
                "pred_payout": result.payout,
                "payout_ok": payout_ok,
                "confidence": round(result.confidence, 3),
                "pred_reasons": result.reasons,
            })
            if sleep:
                time.sleep(sleep)
    finally:
        conn.close()

    base = [r for r in rows if not r["mismatch_type"]]
    mismatch = [r for r in rows if r["mismatch_type"]]
    return {
        "golden_version": payload.get("version"),
        "n": len(rows),
        "route_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "base_route_accuracy": sum(r["route_ok"] for r in base) / len(base) if base else None,
        "mismatch_detected": sum(r["pred_route"] == "escalate" for r in mismatch),
        "mismatch_total": len(mismatch),
        "payout_accuracy_where_truth_pays": _payout_acc(rows),
        "rows": rows,
    }


def _payout_acc(rows) -> float | None:
    payable = [r for r in rows if r["true_payout"] is not None]
    if not payable:
        return None
    return sum(r["payout_ok"] for r in payable) / len(payable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(GOLDEN_DEFAULT))
    parser.add_argument("--sleep", type=float, default=8.0,
                        help="seconds between claims to stay under the Groq TPM cap")
    parser.add_argument("--out", default=str(REPO / "eval" / "arch2_results.json"))
    args = parser.parse_args()

    report = evaluate(Path(args.golden), args.sleep)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\nArch 2 vs golden {report['golden_version']}  (n={report['n']})")
    print(f"  route accuracy overall : {report['route_accuracy']:.0%}")
    print(f"  route accuracy (base)  : {report['base_route_accuracy']:.0%}")
    print(f"  mismatch caught        : {report['mismatch_detected']}/{report['mismatch_total']}  (expected 0 -- Arch 2 is story-blind)")
    pa = report["payout_accuracy_where_truth_pays"]
    print(f"  payout $ accuracy      : {pa:.0%}" if pa is not None else "  payout $ accuracy      : n/a")
    print()
    for r in report["rows"]:
        tag = "MM" if r["mismatch_type"] else "  "
        mark = "ok " if r["route_ok"] else "XX "
        print(f"  {tag} {mark} {r['claim_id']}  true={r['true_route']:<12} pred={r['pred_route']:<12} "
              f"conf={r['confidence']}  payout {r['true_payout']} -> {r['pred_payout']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
