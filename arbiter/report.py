"""Self-contained HTML report generated from results.json."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{--ink:#0f2a1f;--muted:#51645a;--line:#dbe5df;--bg:#fbfdfc;--bad:#a33a1f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans","Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:48px 24px 80px}h1{font-size:2rem;line-height:1.15;margin:0 0 8px}
h2{font-size:1.2rem;margin:44px 0 8px}p{max-width:70ch;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.n{text-align:right}.wrap{overflow-x:auto}
"""


def _t(headers, rows):
    h = "".join(f'<th class="{"n" if i else ""}">{html.escape(x)}</th>' for i, x in enumerate(headers))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def write_report(r: dict, path: Path) -> None:
    tv = _t(["Policy", "Net € per txn", "Conversion", "Fraud bps", "3DS rate"],
            [[html.escape(k), f'{v["net_per_txn"]:.3f}', f'{v["conversion"]:.1%}', f'{v["fraud_bps"]:.1f}', f'{v["three_ds_rate"]:.0%}']
             for k, v in r["true_values"].items()])
    ope = _t(["Policy", "Effective sample", "IPS bias", "SNIPS bias", "DM bias", "DR bias", "IPS RMSE", "DR RMSE", "DR 95% CI coverage"],
             [[html.escape(k), f'{r["effective_sample_share"][k]:.1%}'] + [f'{d[e]["bias"]:+.3f}' for e in ("IPS", "SNIPS", "DM", "DR")]
              + [f'{d["IPS"]["rmse"]:.3f}', f'{d["DR"]["rmse"]:.3f}', f'{r["coverage95"][k]["DR"]:.0%}'] for k, d in r["ope"].items()])
    rk = _t(["Estimator", "Kendall τ with truth", "Picked the true best"],
            [[e, f'{v["kendall_tau_mean"]:.2f}', f'{v["picked_true_best"]:.0%}'] for e, v in r["ranking"].items()])
    mt = _t(["Evaluation day", "Fraud labels known", "Shift, unconstrained", "Shift, constrained", "Shift, rules",
             "Logs where unconstrained looks better"],
            [[row["day"], f'{row["labels_matured"]:.0%}', f'{row["shift_greedy"]:+.3f}', f'{row["shift_constrained"]:+.3f}',
              f'{row["shift_rules"]:+.3f}', f'{row["share_preferring_greedy"]:.0%}'] for row in r["maturity"]["rows"]])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arbiter results</title><style>{CSS}</style></head><body><main>
<h1>Can we trust the offline number before we ship the policy?</h1>
<p>Synthetic payment world with known outcome probabilities. Runtime {r["environment"]["runtime_s"]} s,
{r["setup"]["replicates"]} replicate logs of {r["setup"]["n_eval"]:,} transactions each.</p>
<h2>True value of each policy</h2>{tv}
<h2>Off-policy estimates versus the truth</h2><p>Bias and RMSE in euro per transaction across replicates.</p>{ope}
<h2>Do the estimators rank policies correctly?</h2>{rk}
<h2>Delayed fraud labels</h2><p>DR estimate on each day minus the fully matured estimate. In truth, the unconstrained
policy is {"better" if r["maturity"]["truth_prefers_greedy"] else "worse"} than the constrained one.</p>{mt}
</main></body></html>"""
    Path(path).write_text(doc)
