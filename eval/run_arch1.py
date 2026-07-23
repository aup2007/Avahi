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

from arch1_monolith import vlm_call  # noqa: E402
from eval.judge import judge_reasoning  # noqa: E402

DB_PATH = REPO / "db" / "avahi.db"
IMAGE_DIR = REPO / "CarDD_release" / "CarDD_COCO" / "test2017"
GOLDEN_DEFAULT = REPO / "data" / "golden_set" / "golden_set_v2.json"

PAYOUT_TOL = 0.01


def _payout_matches(pred, truth) -> bool:
    if pred is None or truth is None:
        return pred is None and truth is None
    return abs(pred - truth) <= PAYOUT_TOL


def _is_hallucinated(payout, policy: dict) -> tuple[bool, str | None]:
    # A payout the contract cannot produce, checked only against figures the model
    # was given in its own prompt (status, limits, deductible) -- never against the
    # cost table, which Arch 1 never sees. So this flags the model contradicting
    # its own inputs, not merely mis-estimating the repair.
    if payout is None or payout <= 0:
        return False, None
    if policy["policy_status"] == "lapsed":
        return True, "paid on a lapsed policy"
    if not policy["collision_active"] and not policy["comprehensive_active"]:
        return True, "paid with no active coverage"
    active_limits = [
        limit
        for active, limit in (
            (policy["collision_active"], policy["collision_limit"]),
            (policy["comprehensive_active"], policy["comprehensive_limit"]),
        )
        if active
    ]
    # Most generous reading: the largest active limit, minus the deductible.
    ceiling = max(active_limits) - policy["deductible"]
    if payout > ceiling + PAYOUT_TOL:
        return True, f"payout {payout:.2f} exceeds ceiling {ceiling:.2f} (limit - deductible)"
    return False, None


def _run_once(conn, records, sleep: float, judge: bool) -> list[dict]:
    rows = []
    for rec in records:
        image_path = str(IMAGE_DIR / rec["photo_file"])
        result = vlm_call.decide(conn, rec["claim_id"], image_path, rec.get("claim_story"))
        policy = vlm_call._fetch_policy(conn, rec["customer_id"])

        hallucinated, why = _is_hallucinated(result.payout, policy)
        row = {
            "claim_id": rec["claim_id"],
            "mismatch_type": rec.get("mismatch_type"),
            "true_route": rec["route"],
            "pred_route": result.route,
            "route_ok": result.route == rec["route"],
            "true_payout": rec["payout"],
            "pred_payout": result.payout,
            "payout_ok": _payout_matches(result.payout, rec["payout"]),
            "coverage_type": result.coverage_type,
            "hallucinated_payout": hallucinated,
            "hallucination_reason": why,
            "reasoning": result.reasoning,
            "latency_s": round(result.latency_s, 2),
        }
        if judge:
            row["judge"] = judge_reasoning(policy, result)
            if sleep:
                time.sleep(sleep)
        rows.append(row)
        if sleep:
            time.sleep(sleep)
    return rows


def _rate(rows, key) -> float | None:
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def _payout_stats(rows) -> dict:
    # Reported, but read them with SPEC.md §11 in mind: Arch 1 gets no cost table,
    # so its dollar figure is unanchored by construction. A near-zero exact-match
    # is the expected result, not a surprising one -- the informative column is
    # hallucination rate, which only uses figures the model was handed.
    scored = [r for r in rows if r["true_payout"] is not None and r["pred_payout"] is not None]
    exact = [r for r in rows if r["true_payout"] is not None]
    return {
        "exact_match": (sum(r["payout_ok"] for r in exact) / len(exact)) if exact else None,
        "mae": (sum(abs(r["pred_payout"] - r["true_payout"]) for r in scored) / len(scored))
        if scored
        else None,
        "n_scored_for_mae": len(scored),
    }


def _decision_errors(rows) -> dict:
    # False approve: paid a claim the golden set says should not have been paid
    # (auto_deny) or should have gone to a human (escalate). False deny: refused a
    # claim that was owed. Escalate rows land in one bucket or the other because
    # the monolith has no escalate branch -- which is the finding.
    false_approve = [r for r in rows if r["pred_route"] == "auto_approve" and r["true_route"] != "auto_approve"]
    false_deny = [r for r in rows if r["pred_route"] == "auto_deny" and r["true_route"] == "auto_approve"]
    should_escalate = [r for r in rows if r["true_route"] == "escalate"]
    return {
        "false_approve_rate": len(false_approve) / len(rows) if rows else None,
        "false_deny_rate": len(false_deny) / len(rows) if rows else None,
        "false_approve_ids": [r["claim_id"] for r in false_approve],
        "false_deny_ids": [r["claim_id"] for r in false_deny],
        "escalate_rows_auto_decided": len(should_escalate),
        "escalate_rows_total": len(should_escalate),
    }


def _reproducibility(run_a, run_b) -> dict:
    by_id_b = {r["claim_id"]: r for r in run_b}
    same_route = same_payout = 0
    drift = []
    for a in run_a:
        b = by_id_b.get(a["claim_id"])
        if b is None:
            continue
        route_same = a["pred_route"] == b["pred_route"]
        payout_same = _payout_matches(a["pred_payout"], b["pred_payout"])
        same_route += route_same
        same_payout += payout_same
        if not (route_same and payout_same):
            drift.append({
                "claim_id": a["claim_id"],
                "route": [a["pred_route"], b["pred_route"]],
                "payout": [a["pred_payout"], b["pred_payout"]],
            })
    n = len(run_a)
    return {
        "route_stability": same_route / n if n else None,
        "payout_stability": same_payout / n if n else None,
        "drifted": drift,
    }


def _judge_stats(rows) -> dict | None:
    judged = [r for r in rows if r.get("judge")]
    if not judged:
        return None
    return {
        "n": len(judged),
        "coherent_rate": sum(r["judge"]["coherent"] for r in judged) / len(judged),
        "derivation_shown_rate": sum(r["judge"]["derivation_shown"] for r in judged) / len(judged),
        "incoherent_ids": [r["claim_id"] for r in judged if not r["judge"]["coherent"]],
    }


def evaluate(golden_path: Path, sleep: float, runs: int, judge: bool, limit: int | None) -> dict:
    payload = json.loads(golden_path.read_text())
    records = payload["records"]
    if limit:
        records = records[:limit]
    conn = sqlite3.connect(DB_PATH)

    try:
        all_runs = []
        for i in range(runs):
            # Only the first run is judged -- the judge grades reasoning coherence,
            # which does not need repeating for the reproducibility pass.
            all_runs.append(_run_once(conn, records, sleep, judge and i == 0))
    finally:
        conn.close()

    rows = all_runs[0]
    report = {
        "golden_version": payload.get("version"),
        "n": len(rows),
        "runs": runs,
        "decision_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "payout": _payout_stats(rows),
        "decision_errors": _decision_errors(rows),
        "hallucinated_payout_rate": _rate(rows, "hallucinated_payout"),
        "hallucinated_ids": [
            {"claim_id": r["claim_id"], "reason": r["hallucination_reason"]}
            for r in rows
            if r["hallucinated_payout"]
        ],
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
        "judge": _judge_stats(rows),
        "reproducibility": _reproducibility(all_runs[0], all_runs[1]) if runs > 1 else None,
        "rows": rows,
    }
    if runs > 1:
        report["run_2_rows"] = all_runs[1]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(GOLDEN_DEFAULT))
    parser.add_argument("--sleep", type=float, default=8.0,
                        help="seconds between calls to stay under the Groq TPM cap")
    parser.add_argument("--runs", type=int, default=2,
                        help="run the whole set N times to measure reproducibility (SPEC.md:83)")
    parser.add_argument("--judge", action="store_true", default=True,
                        help="LLM-as-judge audit of reasoning-vs-payout coherence")
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N golden records")
    parser.add_argument("--out", default=str(REPO / "eval" / "arch1_results.json"))
    args = parser.parse_args()

    report = evaluate(Path(args.golden), args.sleep, args.runs, args.judge, args.limit)
    Path(args.out).write_text(json.dumps(report, indent=2))

    err = report["decision_errors"]
    pay = report["payout"]
    print(f"\nArch 1 (monolith) vs golden {report['golden_version']}  (n={report['n']}, runs={report['runs']})")
    print(f"  decision accuracy      : {report['decision_accuracy']:.0%}")
    print(f"  false approve / deny   : {err['false_approve_rate']:.0%} / {err['false_deny_rate']:.0%}")
    print(f"  escalate rows decided  : {err['escalate_rows_auto_decided']}/{err['escalate_rows_total']}  "
          f"(monolith has no escalate branch)")
    em = f"{pay['exact_match']:.0%}" if pay["exact_match"] is not None else "n/a"
    mae = f"${pay['mae']:,.0f}" if pay["mae"] is not None else "n/a"
    print(f"  payout exact-match     : {em}   MAE: {mae}  (no cost table -- unanchored)")
    print(f"  hallucinated payout    : {report['hallucinated_payout_rate']:.0%}")
    if report["judge"]:
        j = report["judge"]
        print(f"  reasoning coherent     : {j['coherent_rate']:.0%}   derivation shown: {j['derivation_shown_rate']:.0%}")
    if report["reproducibility"]:
        rp = report["reproducibility"]
        print(f"  reproducibility        : route {rp['route_stability']:.0%}, payout {rp['payout_stability']:.0%}")
    print(f"  mean latency           : {report['mean_latency_s']:.1f}s")
    print()
    for r in report["rows"]:
        mark = "ok " if r["route_ok"] else "XX "
        flag = " HALLUC" if r["hallucinated_payout"] else ""
        print(f"  {mark} {r['claim_id']}  true={r['true_route']:<12} pred={r['pred_route']:<12} "
              f"payout {r['true_payout']} -> {r['pred_payout']}{flag}")
        if r["hallucination_reason"]:
            print(f"        ! {r['hallucination_reason']}")
        if r.get("judge") and not r["judge"]["coherent"]:
            print(f"        judge: {r['judge']['note']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
