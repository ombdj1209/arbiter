"""Policies, reward models learned from logs, and off-policy estimators.

Estimators (value of a target policy pi from data logged under mu):
  IPS    mean(pi(a|x) / mu(a|x) * r)
  SNIPS  IPS normalised by the mean importance weight
  DM     mean over x of sum_a pi(a|x) q_hat(x, a)            (model only)
  DR     DM + mean(pi(a|x) / mu(a|x) * (r - q_hat(x, a)))    (model plus correction)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .world import ACTIONS, CHARGEBACK_FEE, MARGIN, fee

N_A = len(ACTIONS)


def one_hot(actions: np.ndarray) -> np.ndarray:
    return np.eye(N_A)[actions]


# ------------------------------------------------------------------------------- policies
def rules(c: dict) -> np.ndarray:
    """A typical hand-written production policy: 3DS on high amounts or risky countries, always route A."""
    tds = (c["amount"] > 150) | (c["risk_country"] == 1)
    return one_hot(np.where(tds, 1, 0))


def epsilon(pi: np.ndarray, eps: float) -> np.ndarray:
    return (1 - eps) * pi + eps / N_A


def greedy(q: np.ndarray) -> np.ndarray:
    return one_hot(q.argmax(1))


def constrained(q: np.ndarray, fraud_vol: np.ndarray, lam: float) -> np.ndarray:
    """Lagrangian policy: maximise predicted net revenue minus lam x predicted merchant-liable fraud volume."""
    return one_hot((q - lam * fraud_vol).argmax(1))


def sample(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(len(pi))[:, None]
    return (pi.cumsum(1) < u).sum(1).clip(0, N_A - 1)


# -------------------------------------------------------------------------- reward models
class RewardModel:
    """Decomposed like the business: P(abandon | x, 3DS), P(approved | x, a), P(fraud | approved, x, a).
    Fraud is learned only from transactions whose chargeback window has matured."""

    def __init__(self, seed: int = 0):
        kw = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
        self.ab, self.ap, self.fr = (HistGradientBoostingClassifier(**kw) for _ in range(3))

    @staticmethod
    def _xa(X, actions):
        return np.hstack([X, np.asarray([ACTIONS[a] for a in actions], dtype=float)])

    def fit(self, X, actions, logs, matured: np.ndarray):
        tds = logs["three_ds"] == 1
        self.ab_fitted = bool(tds.any() and logs["abandon"][tds].any())
        if self.ab_fitted:
            self.ab.fit(X[tds], logs["abandon"][tds])
        seen = ~logs["abandon"]
        self.ap.fit(self._xa(X[seen], actions[seen]), logs["approved"][seen])
        lab = logs["approved"] & matured
        self.fr.fit(self._xa(X[lab], actions[lab]), logs["fraud"][lab])
        return self

    def predict(self, X, amount):
        """Returns q_hat (n, 4) expected net revenue, predicted merchant-liable fraud volume (n, 4) and
        predicted approved volume (n, 4)."""
        n = len(X)
        q, fv, av = np.zeros((n, N_A)), np.zeros((n, N_A)), np.zeros((n, N_A))
        ab3 = self.ab.predict_proba(X)[:, 1] if self.ab_fitted else np.zeros(n)
        for j, (rb, tds) in enumerate(ACTIONS):
            xa = self._xa(X, np.full(n, j))
            ab = ab3 * tds
            ap = self.ap.predict_proba(xa)[:, 1]
            fr = self.fr.predict_proba(xa)[:, 1]
            good = MARGIN * amount - fee(amount, rb)
            bad = good if tds else -(1 - MARGIN) * amount - fee(amount, rb) - CHARGEBACK_FEE
            q[:, j] = (1 - ab) * ap * ((1 - fr) * good + fr * bad)
            fv[:, j] = (1 - ab) * ap * fr * amount * (1 - tds)
            av[:, j] = (1 - ab) * ap * amount
        return q, fv, av


# ----------------------------------------------------------------------------- estimators
def estimates(pi, mu, actions, r, q_hat) -> dict:
    idx = np.arange(len(actions))
    w = pi[idx, actions] / mu[idx, actions]
    dm_x = (pi * q_hat).sum(1)
    return {"IPS": float(np.mean(w * r)), "SNIPS": float(np.sum(w * r) / np.sum(w)),
            "DM": float(dm_x.mean()), "DR": float(np.mean(dm_x + w * (r - q_hat[idx, actions])))}


def bootstrap_ci(pi, mu, actions, r, q_hat, rng, b: int = 200) -> dict:
    n = len(actions)
    idx = np.arange(n)
    w = pi[idx, actions] / mu[idx, actions]
    dm_x = (pi * q_hat).sum(1)
    terms = {"IPS": w * r, "DR": dm_x + w * (r - q_hat[idx, actions])}
    out = {}
    for k, t in terms.items():
        bs = [t[rng.integers(0, n, n)].mean() for _ in range(b)]
        out[k] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out
