"""Operating-point drift detector.

The fusion per-class thresholds are FIXED at calibration time. When live
traffic shifts (a new language mix, a new attack style, a new tenant), the
effective false-positive rate at those fixed thresholds silently drifts away
from the budget the thresholds were calibrated for -- and without a monitor
that only surfaces on a manual re-measurement.

This watches the live score distribution in audit.jsonl and estimates the
CURRENT effective FPR at the deployed threshold, plus a PSI drift score
against the calibration-time reference. No raw text, no labels needed —
audit records already carry `semantic_score` + `decision` + `source`.

    python scripts/monitor_operating_point.py
    python scripts/monitor_operating_point.py --audit audit.jsonl --window 5000
    python scripts/monitor_operating_point.py --self-test
    python scripts/monitor_operating_point.py --json-out _evidence/operating_point.json

Output per source + overall:
  ref p50/p90/p95/p99 | live p50/p90/p95/p99 | PSI | est. effective FPR | verdict

Caveats:
  - `decision in (ALLOW, RESTRICT)` is a PROXY for "benign": it excludes
    true-positive BLOCKs but also false-negatives, so the estimate is
    slightly optimistic. Good for a drift TREND, not a substitute for a
    labelled eval.
  - Needs a representative production audit window. On a fresh/dev log it
    will read RED off noise.
"""
from __future__ import annotations
import argparse
import json
import math
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF_SIGNALS = os.path.join(REPO, "_evidence", "suite_signals_fusion.jsonl")


def quantiles(xs, qs=(0.5, 0.9, 0.95, 0.99)):
    if not xs:
        return {q: None for q in qs}
    s = sorted(xs)
    return {q: round(s[min(len(s) - 1, int(q * len(s)))], 4) for q in qs}


def psi(ref, live, bins=10):
    """Population Stability Index between two score samples in [0,1]."""
    if not ref or not live:
        return None
    def hist(xs):
        h = [0] * bins
        for x in xs:
            k = min(bins - 1, int(x * bins))
            h[k] += 1
        return [c / len(xs) for c in h]

    pr, pl = hist(ref), hist(live)
    total = 0.0
    for a, b in zip(pr, pl):
        a = max(a, 1e-6)
        b = max(b, 1e-6)
        total += (b - a) * math.log(b / a)
    return round(total, 4)


def load_reference():
    """Benign fusion scores + the deployed operating point, from the eval suite."""
    if not os.path.exists(REF_SIGNALS):
        raise SystemExit(
            f"reference signals not found: {REF_SIGNALS}\n"
            "run scripts/evaluate_suite.py --fusion --refresh-only-features first, "
            "or point --ref at an existing signals file."
        )
    rows = [json.loads(line) for line in open(REF_SIGNALS, encoding="utf-8")]
    benign = [float(r["fusion_score"]) for r in rows
              if r.get("label") == 0 and r.get("fusion_score") is not None]
    by_src = {}
    for r in rows:
        if r.get("label") == 0 and r.get("fusion_score") is not None:
            by_src.setdefault(r.get("source", "?"), []).append(float(r["fusion_score"]))
    s = sorted(benign)
    thr = s[int(0.95 * len(s))]  # score that puts 5% of benign above it
    return benign, by_src, round(thr, 4)


def load_live(audit_path, window):
    """Recent 'benign-ish' scores from the audit log: decisions NOT flagged as
    attacks (ALLOW / RESTRICT) are the population the FPR is about."""
    if not os.path.exists(audit_path):
        raise SystemExit(f"audit log not found: {audit_path}")
    recs = []
    with open(audit_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "semantic_score" not in r or "decision" not in r:
                continue
            recs.append(r)
    recs = recs[-window:]
    scores = [float(r["semantic_score"]) for r in recs
              if r["decision"] in ("ALLOW", "RESTRICT")]
    by_src = {}
    for r in recs:
        if r["decision"] in ("ALLOW", "RESTRICT"):
            by_src.setdefault(r.get("source", "audit"), []).append(float(r["semantic_score"]))
    return scores, by_src, len(recs)


def eff_fpr(scores, thr):
    if not scores:
        return None
    return round(sum(1 for x in scores if x >= thr) / len(scores), 4)


def verdict(psi_val, efpr, budget=0.05):
    if psi_val is None or efpr is None:
        return "NO DATA"
    if efpr > budget * 1.5 or psi_val > 0.25:
        return "RED (recalibrate)"
    if efpr > budget * 1.1 or psi_val > 0.10:
        return "AMBER (watch)"
    return "GREEN"


def report(ref, live, thr, label, budget=0.05):
    rq, lq = quantiles(ref), quantiles(live)
    p = psi(ref, live)
    ef = eff_fpr(live, thr)
    v = verdict(p, ef, budget)
    print(f"\n[{label}]  n_ref={len(ref)}  n_live={len(live)}  threshold={thr}")
    print(f"  ref  p50={rq[0.5]}  p90={rq[0.9]}  p95={rq[0.95]}  p99={rq[0.99]}")
    print(f"  live p50={lq[0.5]}  p90={lq[0.9]}  p95={lq[0.95]}  p99={lq[0.99]}")
    print(f"  PSI={p}   est. effective FPR at threshold = "
          f"{('%.2f%%' % (ef * 100)) if ef is not None else 'n/a'}  (budget {budget * 100:.0f}%)")
    print(f"  VERDICT: {v}")
    return {"label": label, "psi": p, "eff_fpr": ef, "verdict": v,
            "ref_q": rq, "live_q": lq, "threshold": thr}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default=os.path.join(REPO, "audit.jsonl"))
    ap.add_argument("--window", type=int, default=20000,
                    help="most recent N audit records to consider (default 20000)")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="target benign FPR the thresholds were calibrated for")
    ap.add_argument("--json-out", default=None,
                    help="also write the full report as JSON to this path")
    ap.add_argument("--self-test", action="store_true",
                    help="use the suite as its own 'live' (must be GREEN) + a "
                         "German-heavy slice (must drift RED)")
    args = ap.parse_args()

    ref_all, ref_by_src, thr = load_reference()
    out = []

    if args.self_test:
        rows = [json.loads(line) for line in open(REF_SIGNALS, encoding="utf-8")]
        benign = [r for r in rows if r.get("label") == 0 and r.get("fusion_score") is not None]
        same = [float(r["fusion_score"]) for r in benign]
        de = [float(r["fusion_score"]) for r in benign if r.get("language") == "de"]
        en = [float(r["fusion_score"]) for r in benign if r.get("language") == "en"]
        out.append(report(ref_all, same, thr,
                          "self-test: suite vs itself (expect GREEN)", args.budget))
        out.append(report(en, de, thr,
                          "self-test: EN-benign ref vs DE-benign live (expect drift RED)",
                          args.budget))
    else:
        live_all, live_by_src, n_recs = load_live(args.audit, args.window)
        print(f"audit: {args.audit}  records scanned={n_recs}  "
              f"benign-ish live scores={len(live_all)}")
        out.append(report(ref_all, live_all, thr, "OVERALL", args.budget))
        for src in sorted(set(ref_by_src) & set(live_by_src)):
            out.append(report(ref_by_src[src], live_by_src[src], thr,
                              f"source={src}", args.budget))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"threshold": thr, "budget": args.budget, "results": out}, f, indent=2)
        print(f"\nsaved: {args.json_out}")

    if args.self_test:
        return  # diagnostic mode: the RED German slice is the expected result
    worst = max((r for r in out if r["verdict"].startswith(("RED", "AMBER"))),
                default=None, key=lambda r: r["psi"] or 0)
    raise SystemExit(2 if worst and worst["verdict"].startswith("RED") else 0)


if __name__ == "__main__":
    main()
