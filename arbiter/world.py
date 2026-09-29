"""Synthetic payment world with a known outcome model.

Each transaction gets one of four actions: route A or B, with or without 3-D Secure (3DS).
  - 3DS adds friction: some shoppers abandon.
  - 3DS raises issuer approval and shifts fraud liability to the issuer; most fraudsters fail it.
  - Routes differ in cost and in approval rates per issuer.
  - Fraud is only confirmed when a chargeback arrives, 15 to 120 days later.

Because the true probabilities are known, the exact expected value of any policy can be computed,
which is what lets this lab grade off-policy estimators against the truth instead of against each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ((0, 0), (0, 1), (1, 0), (1, 1))            # (route B?, 3DS?)
ACTION_NAMES = ("A", "A+3DS", "B", "B+3DS")
N_ISSUERS = 8
MARGIN = 0.20                                          # merchant gross margin on a sale
CHARGEBACK_FEE = 15.0


@dataclass(frozen=True)
class Params:
    issuer_base: tuple = (1.9, 1.6, 2.2, 1.2, 1.8, 1.4, 2.0, 1.0)
    route_b_affinity: tuple = (0.7, -0.4, 0.5, 0.9, -0.5, 0.6, -0.3, 1.0)   # logit bonus for route B per issuer
    three_ds_approval: float = 0.6
    fraud_fails_3ds: float = 0.85


P = Params()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def contexts(n: int, rng: np.random.Generator) -> dict:
    amt = np.round(rng.lognormal(np.log(60), 0.9, n), 2).clip(3, 3000)
    return {
        "amount": amt,
        "risk_country": (rng.random(n) < 0.12).astype(float),
        "mobile": (rng.random(n) < 0.58).astype(float),
        "returning": (rng.random(n) < 0.55).astype(float),
        "night": (rng.random(n) < 0.18).astype(float),
        "credit": (rng.random(n) < 0.6).astype(float),
        "velocity": rng.poisson(0.6, n).astype(float),
        "issuer": rng.integers(0, N_ISSUERS, n),
        "day": rng.uniform(0, 30, n),                   # when it happened within the logging window
    }


def features(c: dict) -> np.ndarray:
    onehot = np.eye(N_ISSUERS)[c["issuer"]]
    base = np.column_stack([np.log(c["amount"]), c["risk_country"], c["mobile"], c["returning"], c["night"],
                            c["credit"], c["velocity"]])
    return np.hstack([base, onehot])


def fee(amount, route_b):
    return np.where(route_b, 0.0030 * amount + 0.05, 0.0020 * amount + 0.10)


def truth(c: dict):
    """True probabilities for every action. Returns dict of (n, 4) arrays."""
    n = len(c["amount"])
    f = sigmoid(-4.6 + 1.3 * c["risk_country"] + 0.9 * (1 - c["returning"]) + 0.45 * np.log(c["amount"] / 50)
                + 0.5 * (c["velocity"] > 2) + 0.4 * c["night"] - 0.2 * c["credit"])
    ib = np.asarray(P.issuer_base)[c["issuer"]]
    aff = np.asarray(P.route_b_affinity)[c["issuer"]]
    out = {k: np.zeros((n, 4)) for k in ("abandon", "appr_legit", "appr_fraud", "reward", "conv", "fraud_loss_vol", "appr_vol")}
    for j, (rb, tds) in enumerate(ACTIONS):
        ab = tds * (0.05 + 0.09 * c["mobile"] + 0.04 * (1 - c["returning"]) + 0.03 * (c["amount"] < 25))
        z = ib + rb * aff + tds * P.three_ds_approval - 0.35 * c["risk_country"] - 0.12 * np.log(c["amount"] / 50)
        pl = sigmoid(z)
        pf = sigmoid(z - 1.6) * (1 - tds * P.fraud_fails_3ds)
        fe = fee(c["amount"], rb)
        good = MARGIN * c["amount"] - fe
        bad = good if tds else -(1 - MARGIN) * c["amount"] - fe - CHARGEBACK_FEE   # liability shift under 3DS
        out["abandon"][:, j] = ab
        out["appr_legit"][:, j] = pl
        out["appr_fraud"][:, j] = pf
        out["conv"][:, j] = (1 - ab) * ((1 - f) * pl + f * pf)
        out["reward"][:, j] = (1 - ab) * ((1 - f) * pl * good + f * pf * bad)
        out["appr_vol"][:, j] = out["conv"][:, j] * c["amount"]
        out["fraud_loss_vol"][:, j] = (1 - ab) * f * pf * c["amount"] * (1 - tds)   # merchant-liable fraud volume
    out["fraud_p"] = f
    return out


def policy_value(pi: np.ndarray, t: dict) -> dict:
    """Exact expected metrics of a stochastic policy pi (n, 4) under the true model."""
    rev = float((pi * t["reward"]).sum(1).mean())
    conv = float((pi * t["conv"]).sum(1).mean())
    appr = (pi * t["appr_vol"]).sum()
    fraud = (pi * t["fraud_loss_vol"]).sum()
    tds = float(pi[:, [1, 3]].sum(1).mean())
    return {"net_per_txn": rev, "conversion": conv, "fraud_bps": float(1e4 * fraud / max(appr, 1e-9)), "three_ds_rate": tds}


def log_outcomes(c: dict, t: dict, actions: np.ndarray, rng: np.random.Generator) -> dict:
    """Sample realised outcomes for the logged actions, including a chargeback delay for fraud."""
    n = len(actions)
    idx = np.arange(n)
    fraud = rng.random(n) < t["fraud_p"]
    abandon = rng.random(n) < t["abandon"][idx, actions]
    p_appr = np.where(fraud, t["appr_fraud"][idx, actions], t["appr_legit"][idx, actions])
    approved = (~abandon) & (rng.random(n) < p_appr)
    rb = np.asarray([ACTIONS[a][0] for a in actions])
    tds = np.asarray([ACTIONS[a][1] for a in actions])
    fe = fee(c["amount"], rb)
    good = MARGIN * c["amount"] - fe
    bad = np.where(tds == 1, good, -(1 - MARGIN) * c["amount"] - fe - CHARGEBACK_FEE)
    reward = np.where(approved, np.where(fraud, bad, good), 0.0)
    delay = rng.uniform(15, 120, n)
    return {"fraud": fraud, "abandon": abandon, "approved": approved, "reward": reward, "three_ds": tds,
            "chargeback_day": c["day"] + delay, "good_reward": np.where(approved, good, 0.0)}
