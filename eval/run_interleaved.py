import argparse
import json
import sqlite3
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval import run_arch1 as ra1  # noqa: E402  (also loads .env)
from eval.judge import judge_reasoning  # noqa: E402
from arch1_monolith import vlm_call  # noqa: E402
from arch2_split import pipeline  # noqa: E402
from arch3_agent import agent  # noqa: E402

DB_PATH = REPO / "db" / "avahi.db"
IMAGE_DIR = REPO / "CarDD_release" / "CarDD_COCO" / "test2017"
GOLDEN_DEFAULT = REPO / "data" / "golden_set" / "golden_set_v2.json"


def _arch1_row(conn, rec, image_path: str) -> tuple[dict, object, dict]:
    result = vlm_call.decide(conn, rec["claim_id"], image_path)
    policy = vlm_call._fetch_policy(conn, rec["customer_id"])
    hallucinated, why = ra1._is_hallucinated(result.payout, policy)
    row = {
        "claim_id": rec["claim_id"],
        "mismatch_type": rec.get("mismatch_type"),
        "true_route": rec["route"],
        "pred_route": result.route,
        "route_ok": result.route == rec["route"],
        "true_payout": rec["payout"],
        "pred_payout": result.payout,
        "payout_ok": ra1._payout_matches(result.payout, rec["payout"]),
        "coverage_type": result.coverage_type,
        "hallucinated_payout": hallucinated,
        "hallucination_reason": why,
        "reasoning": result.reasoning,
        "latency_s": round(result.latency_s, 2),
    }
    return row, result, policy


def _arch2_row(conn, rec, image_path: str) -> dict:
    t0 = time.monotonic()
    result = pipeline.run_claim(conn, rec["claim_id"], image_path)
    latency_s = time.monotonic() - t0
    return {
        "claim_id": rec["claim_id"],
        "mismatch_type": rec.get("mismatch_type"),
        "true_route": rec["route"],
        "pred_route": result.route,
        "route_ok": result.route == rec["route"],
        "true_payout": rec["payout"],
        "pred_payout": result.payout,
        "payout_ok": ra1._payout_matches(result.payout, rec["payout"]),
        "confidence": round(result.confidence, 3),
        "pred_reasons": result.reasons,
        "latency_s": round(latency_s, 2),
    }


def _arch3_row(conn, rec, image_path: str) -> dict:
    t0 = time.monotonic()
    result = agent.run_claim(conn, rec["claim_id"], image_path, rec.get("claim_story"))
    latency_s = time.monotonic() - t0
    return {
        "claim_id": rec["claim_id"],
        "mismatch_type": rec.get("mismatch_type"),
        "true_route": rec["route"],
        "pred_route": result.route,
        "route_ok": result.route == rec["route"],
        "true_payout": rec["payout"],
        "pred_payout": result.payout,
        "payout_ok": ra1._payout_matches(result.payout, rec["payout"]),
        "confidence": round(result.confidence, 3),
        "tool_calls": result.tool_calls,
        "replans": result.replans,
        "pred_reasons": result.reasons,
        "latency_s": round(latency_s, 2),
        "trajectory": [f"{t['node']}: {t['summary']}" for t in result.trajectory],
    }


def _split_base_mismatch(rows):
    base = [r for r in rows if not r["mismatch_type"]]
    mismatch = [r for r in rows if r["mismatch_type"]]
    return base, mismatch


def _payout_acc(rows):
    payable = [r for r in rows if r["true_payout"] is not None]
    return sum(r["payout_ok"] for r in payable) / len(payable) if payable else None


def _escalation_pr(rows):
    pred_esc = [r for r in rows if r["pred_route"] == "escalate"]
    true_esc = [r for r in rows if r["true_route"] == "escalate"]
    tp = sum(1 for r in pred_esc if r["true_route"] == "escalate")
    return {
        "precision": tp / len(pred_esc) if pred_esc else None,
        "recall": tp / len(true_esc) if true_esc else None,
        "true_escalate": len(true_esc),
        "pred_escalate": len(pred_esc),
    }


def _arch1_report(version, rows, run2_rows, runs: int) -> dict:
    report = {
        "golden_version": version,
        "n": len(rows),
        "runs": runs,
        "decision_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "payout": ra1._payout_stats(rows),
        "decision_errors": ra1._decision_errors(rows),
        "hallucinated_payout_rate": ra1._rate(rows, "hallucinated_payout"),
        "hallucinated_ids": [
            {"claim_id": r["claim_id"], "reason": r["hallucination_reason"]}
            for r in rows
            if r["hallucinated_payout"]
        ],
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
        "judge": ra1._judge_stats(rows),
        "reproducibility": ra1._reproducibility(rows, run2_rows) if runs > 1 else None,
        "rows": rows,
    }
    if runs > 1:
        report["run_2_rows"] = run2_rows
    return report


def _arch23_report(version, rows, escalation: bool) -> dict:
    base, mismatch = _split_base_mismatch(rows)
    report = {
        "golden_version": version,
        "n": len(rows),
        "route_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "base_route_accuracy": sum(r["route_ok"] for r in base) / len(base) if base else None,
        "mismatch_detected": sum(r["pred_route"] == "escalate" for r in mismatch),
        "mismatch_total": len(mismatch),
        "payout_accuracy_where_truth_pays": _payout_acc(rows),
        "rows": rows,
    }
    if escalation:
        report["escalation"] = _escalation_pr(rows)
    return report


def _save_partial(out_dir: Path, prefix: str, version, state: dict) -> None:
    for arch, rows in (("arch1", state["a1"]), ("arch2", state["a2"]), ("arch3", state["a3"])):
        path = out_dir / f"{prefix}{arch}_results.json"
        path.write_text(json.dumps({
            "golden_version": version,
            "partial": True,
            "n": len(rows),
            "rows": rows,
            "run_2_rows": state["a1_run2"] if arch == "arch1" else None,
        }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(GOLDEN_DEFAULT))
    parser.add_argument("--sleep", type=float, default=8.0,
                        help="seconds between calls to stay under the Groq TPM cap")
    parser.add_argument("--runs", type=int, default=2,
                        help="arch1 passes per claim (2 = reproducibility)")
    parser.add_argument("--judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefix", default="", help="output filename prefix, e.g. smoke_")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    payload = json.loads(golden_path.read_text())
    version = payload.get("version")
    records = payload["records"]
    if args.limit:
        records = records[: args.limit]

    out_dir = REPO / "eval"
    state = {"a1": [], "a1_run2": [], "a2": [], "a3": []}
    conn = sqlite3.connect(DB_PATH)
    t_start = time.monotonic()

    try:
        for i, rec in enumerate(records, 1):
            image_path = str(IMAGE_DIR / rec["photo_file"])
            print(f"[{i}/{len(records)}] {rec['claim_id']}", flush=True)

            row1, result1, policy = _arch1_row(conn, rec, image_path)
            time.sleep(args.sleep)
            if args.runs > 1:
                row1b, _, _ = _arch1_row(conn, rec, image_path)
                state["a1_run2"].append(row1b)
                time.sleep(args.sleep)
            if args.judge:
                row1["judge"] = judge_reasoning(policy, result1)
                time.sleep(args.sleep)
            state["a1"].append(row1)
            print(f"    arch1 pred={row1['pred_route']} payout={row1['pred_payout']}", flush=True)

            row2 = _arch2_row(conn, rec, image_path)
            state["a2"].append(row2)
            print(f"    arch2 pred={row2['pred_route']} payout={row2['pred_payout']}", flush=True)
            time.sleep(args.sleep)

            row3 = _arch3_row(conn, rec, image_path)
            state["a3"].append(row3)
            print(f"    arch3 pred={row3['pred_route']} payout={row3['pred_payout']} "
                  f"[{row3['tool_calls']} calls, {row3['replans']} replans]", flush=True)
            time.sleep(args.sleep)

            _save_partial(out_dir, args.prefix, version, state)
    except Exception:
        _save_partial(out_dir, args.prefix, version, state)
        print("\nABORTED -- partial rows saved:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()

    reports = {
        "arch1": _arch1_report(version, state["a1"], state["a1_run2"], args.runs),
        "arch2": _arch23_report(version, state["a2"], escalation=False),
        "arch3": _arch23_report(version, state["a3"], escalation=True),
    }
    for arch, report in reports.items():
        path = out_dir / f"{args.prefix}{arch}_results.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"wrote {path}", flush=True)

    elapsed = time.monotonic() - t_start
    print(f"\ndone: {len(records)} claims x 3 archs in {elapsed / 60:.1f} min", flush=True)
    print(f"  arch1 decision accuracy : {reports['arch1']['decision_accuracy']:.0%}", flush=True)
    print(f"  arch2 route accuracy    : {reports['arch2']['route_accuracy']:.0%}", flush=True)
    print(f"  arch3 route accuracy    : {reports['arch3']['route_accuracy']:.0%}", flush=True)


if __name__ == "__main__":
    main()
