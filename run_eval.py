"""Reproduces every number in the README.

    python run_eval.py            # full, a few minutes on one core
    python run_eval.py --quick
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from arbiter.lab import (N_A, RewardModel, bootstrap_ci, constrained, epsilon, estimates, greedy, one_hot, rules,
                         sample)
from arbiter.report import write_report
from arbiter.world import ACTION_NAMES, contexts, features, log_outcomes, policy_value, truth

OUT = Path(__file__).resolve().parent / "results"
EPS = 0.10
FRAUD_BUDGET_BPS = 12.0
FULL_MATURITY_DAY = 150.0


def logged(n, rng):
    c = contexts(n, rng)
    t = truth(c)
    mu = epsilon(rules(c), EPS)
    a = sample(mu, rng)
    return c, t, mu, a, log_outcomes(c, t, a, rng)


def observed_reward(logs, day):
    """What the books say on `day`: fraud without a chargeback yet still looks like a good sale."""
    known = logs["chargeback_day"] <= day
    return np.where(logs["fraud"] & ~known, logs["good_reward"], logs["reward"]), known


def cross_fit_q(X, c, a, logs, matured, seed):
    """Two-fold cross-fitting: each half is scored by a model trained on the other half."""
    n = len(a)
    half = np.arange(n) % 2 == 0
    q, fv = np.zeros((n, N_A)), np.zeros((n, N_A))
    for fold in (half, ~half):
        sub = {k: v[~fold] for k, v in logs.items()}
        m = RewardModel(seed).fit(X[~fold], a[~fold], sub, matured[~fold])
        q[fold], fv[fold], _ = m.predict(X[fold], c["amount"][fold])
    return q, fv


def tune_lambda(q, fv, av, budget):
    """Smallest penalty whose *predicted* merchant-liable fraud stays under budget."""
    for lam in np.concatenate([[0], np.geomspace(0.01, 50, 60)]):
        pi = constrained(q, fv, lam)
        if 1e4 * (pi * fv).sum() / max((pi * av).sum(), 1e-9) <= budget:
            return float(lam)
    return 50.0


def main(quick):
    t0 = time.time()
    rng = np.random.default_rng(0)
    n_train, n_eval, reps = (30_000, 20_000, 4) if quick else (80_000, 40_000, 10)

    # 1. learn target policies from one log
    c_tr, t_tr, mu_tr, a_tr, l_tr = logged(n_train, rng)
    X_tr = features(c_tr)
    policy_model = RewardModel(1).fit(X_tr, a_tr, l_tr, l_tr["chargeback_day"] <= FULL_MATURITY_DAY)
    q_tr, fv_tr, av_tr = policy_model.predict(X_tr, c_tr["amount"])
    lam = tune_lambda(q_tr, fv_tr, av_tr, FRAUD_BUDGET_BPS)

    def targets(c, X):
        q, fv, _ = policy_model.predict(X, c["amount"])
        n = len(X)
        return {
            "rules (production)": rules(c),
            "never 3DS, route A": one_hot(np.zeros(n, int)),
            "always 3DS, route A": one_hot(np.ones(n, int)),
            "model greedy (unconstrained)": greedy(q),
            f"model constrained (λ={lam:.2f})": constrained(q, fv, lam),
        }

    # 2. true value of every policy, plus the true Pareto frontier
    c_big = contexts(200_000, np.random.default_rng(7))
    t_big = truth(c_big)
    X_big = features(c_big)
    true_vals = {k: policy_value(pi, t_big) for k, pi in targets(c_big, X_big).items()}
    true_vals["oracle best (knows the truth)"] = policy_value(one_hot(t_big["reward"].argmax(1)), t_big)
    frontier = []
    for lam_true in np.concatenate([[0], np.geomspace(0.01, 50, 30)]):
        score = t_big["reward"] - lam_true * t_big["fraud_loss_vol"]
        frontier.append({"lambda": float(lam_true), **policy_value(one_hot(score.argmax(1)), t_big)})

    # 3. off-policy evaluation: replicate fresh logs, estimate each target, compare with the truth
    names = [k for k in true_vals if not k.startswith("oracle")]
    err = {k: {e: [] for e in ("IPS", "SNIPS", "DM", "DR")} for k in names}
    cover = {k: {"IPS": 0, "DR": 0} for k in names}
    ess = {k: [] for k in names}
    rank_tau = {e: [] for e in ("IPS", "SNIPS", "DM", "DR")}
    days = (45, 90, 150)
    g_name, c_name = "model greedy (unconstrained)", f"model constrained (λ={lam:.2f})"
    mat = {d: {"shift_greedy": [], "shift_constrained": [], "shift_rules": [], "prefers_greedy": [], "matured": []} for d in days}
    best_pick = {e: 0 for e in ("IPS", "SNIPS", "DM", "DR")}
    for rep in range(reps):
        r_rng = np.random.default_rng(100 + rep)
        c, t, mu, a, logs = logged(n_eval, r_rng)
        X = features(c)
        r, matured = observed_reward(logs, FULL_MATURITY_DAY)
        q, _ = cross_fit_q(X, c, a, logs, matured, rep)
        tg = targets(c, X)
        truths, ests = {}, {e: {} for e in rank_tau}
        for k, pi in tg.items():
            v = policy_value(pi, t)["net_per_txn"]
            truths[k] = v
            est = estimates(pi, mu, a, r, q)
            for e, x in est.items():
                err[k][e].append(x - v)
                ests[e][k] = x
            ci = bootstrap_ci(pi, mu, a, r, q, r_rng, b=100 if quick else 200)
            for e, (lo, hi) in ci.items():
                cover[k][e] += lo <= v <= hi
            w = pi[np.arange(len(a)), a] / mu[np.arange(len(a)), a]
            ess[k].append(float(w.sum() ** 2 / (w ** 2).sum() / len(a)))
        order = sorted(truths, key=truths.get)
        for e in rank_tau:
            rank_tau[e].append(kendalltau([truths[k] for k in order], [ests[e][k] for k in order]).statistic)
            best_pick[e] += max(ests[e], key=ests[e].get) == max(truths, key=truths.get)
        # delayed labels: re-estimate with only the chargebacks known on each earlier day
        final = {k: ests["DR"][k] for k in (g_name, c_name, "rules (production)")}
        for d in days:
            r_d, known = observed_reward(logs, d)
            q_d = q if d == FULL_MATURITY_DAY else cross_fit_q(X, c, a, logs, known, rep)[0]
            e_d = {k: estimates(tg[k], mu, a, r_d, q_d)["DR"] for k in final}
            mat[d]["shift_greedy"].append(e_d[g_name] - final[g_name])
            mat[d]["shift_constrained"].append(e_d[c_name] - final[c_name])
            mat[d]["shift_rules"].append(e_d["rules (production)"] - final["rules (production)"])
            mat[d]["prefers_greedy"].append(e_d[g_name] > e_d[c_name])
            mat[d]["matured"].append(float(np.mean(known[logs["fraud"]])))
        print(f"replicate {rep + 1}/{reps} done")

    maturity = [{"day": d, "labels_matured": float(np.mean(v["matured"])),
                 "shift_greedy": float(np.mean(v["shift_greedy"])), "shift_constrained": float(np.mean(v["shift_constrained"])),
                 "shift_rules": float(np.mean(v["shift_rules"])), "share_preferring_greedy": float(np.mean(v["prefers_greedy"]))}
                for d, v in mat.items()]
    truth_prefers_greedy = true_vals[g_name]["net_per_txn"] > true_vals[c_name]["net_per_txn"]
    ope = {k: {e: {"bias": float(np.mean(v)), "rmse": float(np.sqrt(np.mean(np.square(v)))), "sd": float(np.std(v))}
               for e, v in d.items()} for k, d in err.items()}
    coverage = {k: {e: n / reps for e, n in d.items()} for k, d in cover.items()}

    # 5. decision latency: one transaction, four actions scored
    lat = []
    for i in range(300):
        s = time.perf_counter()
        q1, fv1, _ = policy_model.predict(X_big[i:i + 1], c_big["amount"][i:i + 1])
        constrained(q1, fv1, lam)
        lat.append((time.perf_counter() - s) * 1e3)

    res = {"environment": {"python": platform.python_version(), "machine": platform.machine(),
                           "runtime_s": round(time.time() - t0, 1), "quick": quick},
           "setup": {"n_train": n_train, "n_eval": n_eval, "replicates": reps, "epsilon": EPS,
                     "fraud_budget_bps": FRAUD_BUDGET_BPS, "lambda": lam, "actions": ACTION_NAMES},
           "true_values": true_vals, "frontier": frontier, "ope": ope, "coverage95": coverage,
           "effective_sample_share": {k: float(np.mean(v)) for k, v in ess.items()},
           "ranking": {e: {"kendall_tau_mean": float(np.mean(v)), "picked_true_best": best_pick[e] / reps}
                       for e, v in rank_tau.items()},
           "maturity": {"rows": maturity, "truth_prefers_greedy": bool(truth_prefers_greedy)},
           "decision_latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99))}}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    write_report(res, OUT / "report.html")
    for k, v in true_vals.items():
        print(f"{k:34s} net/txn €{v['net_per_txn']:.3f}  conv {v['conversion']:.3f}  fraud {v['fraud_bps']:.1f} bps  3DS {v['three_ds_rate']:.2f}")
    for k, d in ope.items():
        print(k, {e: f"bias {x['bias']:+.3f} rmse {x['rmse']:.3f}" for e, x in d.items()}, coverage[k])
    print("ranking", res["ranking"])
    for row in maturity:
        print(row)
    print("latency", res["decision_latency_ms"], f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
