import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval"

ARCH_LABELS = {
    "arch1": "Arch 1 (monolith)",
    "arch2": "Arch 2 (split pipeline)",
    "arch3": "Arch 3 (PEV agent)",
}


def common_metrics(rows: list[dict]) -> dict:
    # One implementation, applied identically to every architecture's rows.
    # A structurally-impossible metric (e.g. Arch 1 escalation recall) still gets
    # computed the same way -- the bad number IS the comparison's finding.
    n = len(rows)
    base = [r for r in rows if not r.get("mismatch_type")]
    mismatch = [r for r in rows if r.get("mismatch_type")]
    payable = [r for r in rows if r["true_payout"] is not None]
    pred_esc = [r for r in rows if r["pred_route"] == "escalate"]
    true_esc = [r for r in rows if r["true_route"] == "escalate"]
    tp = sum(1 for r in pred_esc if r["true_route"] == "escalate")
    false_approve = [r for r in rows if r["pred_route"] == "auto_approve" and r["true_route"] != "auto_approve"]
    false_deny = [r for r in rows if r["pred_route"] == "auto_deny" and r["true_route"] == "auto_approve"]
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    return {
        "n": n,
        "route_accuracy": sum(r["route_ok"] for r in rows) / n if n else None,
        "base_route_accuracy": sum(r["route_ok"] for r in base) / len(base) if base else None,
        "payout_exact_match_where_truth_pays": (
            sum(r["payout_ok"] for r in payable) / len(payable) if payable else None
        ),
        "mismatch_caught": sum(r["pred_route"] == "escalate" for r in mismatch),
        "mismatch_total": len(mismatch),
        "escalation_precision": tp / len(pred_esc) if pred_esc else None,
        "escalation_recall": tp / len(true_esc) if true_esc else None,
        "false_approve_rate": len(false_approve) / n if n else None,
        "false_deny_rate": len(false_deny) / n if n else None,
        "false_approve_ids": [r["claim_id"] for r in false_approve],
        "false_deny_ids": [r["claim_id"] for r in false_deny],
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
    }


def diagnostics(arch: str, report: dict) -> dict:
    rows = report["rows"]
    if arch == "arch1":
        out = {
            "hallucinated_payout_rate": report.get("hallucinated_payout_rate"),
            "hallucinated_ids": report.get("hallucinated_ids"),
            "payout_mae_unanchored": (report.get("payout") or {}).get("mae"),
        }
        if report.get("reproducibility"):
            rp = report["reproducibility"]
            out["route_stability"] = rp["route_stability"]
            out["payout_stability"] = rp["payout_stability"]
        if report.get("judge"):
            out["reasoning_coherent_rate"] = report["judge"]["coherent_rate"]
            out["derivation_shown_rate"] = report["judge"]["derivation_shown_rate"]
        return out
    if arch == "arch3":
        return {
            "mean_tool_calls": sum(r["tool_calls"] for r in rows) / len(rows) if rows else None,
            "max_tool_calls": max((r["tool_calls"] for r in rows), default=None),
            "mean_replans": sum(r["replans"] for r in rows) / len(rows) if rows else None,
        }
    return {}


def _fmt_pct(v) -> str:
    return f"{v:.0%}" if v is not None else "n/a"


def _fmt_s(v) -> str:
    return f"{v:.1f}s" if v is not None else "n/a"


def _markdown(comparison: dict) -> str:
    archs = ["arch1", "arch2", "arch3"]
    m = {a: comparison["common"][a] for a in archs}
    d = comparison["diagnostics"]
    lines = [
        f"# Architecture comparison -- golden {comparison['golden_version']} "
        f"(n={m['arch1']['n']})",
        "",
        "Same metric, same formula, all three architectures. A bad cell where an",
        "architecture structurally cannot do the task is a finding, not a gap.",
        "",
        "| Metric | " + " | ".join(ARCH_LABELS[a] for a in archs) + " |",
        "|---|---|---|---|",
    ]
    rows_spec = [
        ("Route accuracy (overall)", "route_accuracy", _fmt_pct),
        ("Route accuracy (base 10)", "base_route_accuracy", _fmt_pct),
        ("Payout exact-match (where truth pays)", "payout_exact_match_where_truth_pays", _fmt_pct),
        ("Escalation precision", "escalation_precision", _fmt_pct),
        ("Escalation recall", "escalation_recall", _fmt_pct),
        ("False approve rate", "false_approve_rate", _fmt_pct),
        ("False deny rate", "false_deny_rate", _fmt_pct),
        ("Mean latency / claim", "mean_latency_s", _fmt_s),
    ]
    for label, key, fmt in rows_spec:
        lines.append(f"| {label} | " + " | ".join(fmt(m[a][key]) for a in archs) + " |")
    lines.append(
        "| Story-mismatch caught | "
        + " | ".join(f"{m[a]['mismatch_caught']}/{m[a]['mismatch_total']}" for a in archs)
        + " |"
    )
    lines += ["", "## Per-architecture diagnostics", ""]
    for a in archs:
        if not d.get(a):
            continue
        lines.append(f"### {ARCH_LABELS[a]}")
        for k, v in d[a].items():
            if isinstance(v, float):
                v = f"{v:.0%}" if 0 <= v <= 1 and k.endswith(("rate", "stability")) else f"{v:,.2f}"
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", help="results filename prefix, e.g. smoke_")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    reports = {}
    for arch in ("arch1", "arch2", "arch3"):
        path = EVAL / f"{args.prefix}{arch}_results.json"
        if not path.exists():
            print(f"missing {path} -- run the evals first", file=sys.stderr)
            sys.exit(1)
        reports[arch] = json.loads(path.read_text())
        if reports[arch].get("partial"):
            print(f"warning: {path} is a partial (aborted) run", file=sys.stderr)

    comparison = {
        "golden_version": reports["arch1"].get("golden_version"),
        "common": {arch: common_metrics(r["rows"]) for arch, r in reports.items()},
        "diagnostics": {arch: diagnostics(arch, r) for arch, r in reports.items()},
    }

    out_json = Path(args.out_json) if args.out_json else EVAL / f"{args.prefix}comparison.json"
    out_md = Path(args.out_md) if args.out_md else EVAL / f"{args.prefix}comparison.md"
    out_json.write_text(json.dumps(comparison, indent=2))
    out_md.write_text(_markdown(comparison))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print()
    print(_markdown(comparison))


if __name__ == "__main__":
    main()
