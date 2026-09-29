import numpy as np
import pytest

from arbiter.lab import N_A, RewardModel, constrained, epsilon, estimates, greedy, one_hot, rules, sample
from arbiter.world import contexts, features, log_outcomes, policy_value, truth
from run_eval import FULL_MATURITY_DAY, cross_fit_q, observed_reward, tune_lambda


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(5)
    c = contexts(60_000, rng)
    return c, truth(c), rng


def test_exact_value_matches_monte_carlo(world):
    c, t, rng = world
    pi = rules(c)
    a = sample(pi, rng)
    logs = log_outcomes(c, t, a, rng)
    mc = logs["reward"].mean()
    se = logs["reward"].std() / np.sqrt(len(a))
    assert abs(mc - policy_value(pi, t)["net_per_txn"]) < 4 * se


def test_sampling_follows_policy(world):
    c, t, rng = world
    pi = epsilon(rules(c), 0.2)
    a = sample(pi, rng)
    assert np.allclose(np.bincount(a, minlength=N_A) / len(a), pi.mean(0), atol=0.01)


def test_three_ds_shifts_liability_and_stops_fraud(world):
    c, t, _ = world
    assert np.all(t["fraud_loss_vol"][:, [1, 3]] == 0)
    assert np.all(t["appr_fraud"][:, 1] < t["appr_fraud"][:, 0])
    assert np.all(t["abandon"][:, 0] == 0) and np.all(t["abandon"][:, 1] > 0)


def test_on_policy_ips_is_the_sample_mean(world):
    c, t, rng = world
    mu = epsilon(rules(c), 0.1)
    a = sample(mu, rng)
    r = log_outcomes(c, t, a, rng)["reward"]
    e = estimates(mu, mu, a, r, np.zeros((len(a), N_A)))
    assert e["IPS"] == pytest.approx(r.mean()) and e["DR"] == pytest.approx(r.mean()) and e["DM"] == 0


def test_dr_with_perfect_model_has_tiny_error(world):
    c, t, rng = world
    mu = epsilon(rules(c), 0.1)
    a = sample(mu, rng)
    r = log_outcomes(c, t, a, rng)["reward"]
    pi = one_hot(t["reward"].argmax(1))
    true = policy_value(pi, t)["net_per_txn"]
    e = estimates(pi, mu, a, r, t["reward"])
    assert abs(e["DM"] - true) < 1e-9
    assert abs(e["DR"] - true) < abs(e["IPS"] - true) + 0.05


def test_unmatured_fraud_looks_like_a_good_sale(world):
    c, t, rng = world
    a = sample(rules(c), rng)
    logs = log_outcomes(c, t, a, rng)
    early, known = observed_reward(logs, 40.0)
    hidden = logs["fraud"] & logs["approved"] & ~known & (logs["three_ds"] == 0)
    assert hidden.any()
    assert np.all(early[hidden] > logs["reward"][hidden])
    late, known_all = observed_reward(logs, FULL_MATURITY_DAY)
    assert known_all.all() and np.array_equal(late, logs["reward"])


def test_constrained_with_zero_penalty_is_greedy():
    q = np.random.default_rng(0).normal(size=(100, N_A))
    fv = np.abs(q)
    assert np.array_equal(constrained(q, fv, 0.0), greedy(q))


def test_reward_model_and_lambda_tuning_meet_the_predicted_budget():
    rng = np.random.default_rng(3)
    c = contexts(20_000, rng)
    t = truth(c)
    mu = epsilon(rules(c), 0.1)
    a = sample(mu, rng)
    logs = log_outcomes(c, t, a, rng)
    X = features(c)
    m = RewardModel(0).fit(X, a, logs, np.ones(len(a), bool))
    q, fv, av = m.predict(X, c["amount"])
    lam = tune_lambda(q, fv, av, 12.0)
    pi = constrained(q, fv, lam)
    assert 1e4 * (pi * fv).sum() / (pi * av).sum() <= 12.0
    qc, _ = cross_fit_q(X, c, a, logs, np.ones(len(a), bool), 0)
    assert qc.shape == (len(a), N_A) and np.isfinite(qc).all()
