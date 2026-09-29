# Arbiter

**Can we trust the offline number before we ship the payment policy?** A payment decision lab. A synthetic payment world with *known* outcome probabilities (issuer approval, 3-D Secure friction and liability shift, routing cost, fraud with delayed chargebacks) lets every off-policy estimate be graded against the exact truth, not against another estimate.

It learns an unconstrained and a fraud-constrained policy from logged traffic, then measures how well IPS, SNIPS, direct-method and doubly robust estimators value and rank them, and what happens when fraud labels have not matured yet.

All transactions are synthetic. Nothing here claims to reproduce Adyen's data, models or results.

## Why this exists

Adyen describes its payment AI as a connected decision chain covering risk, authentication, authorisation, routing, retries and cost, optimised together and evaluated counterfactually. Its Uplift product balances conversion, fraud and cost, and Adyen has said it sometimes advises merchants to accept more fraud for a better bottom line. Every one of those decisions depends on one question: **how much can you trust an offline estimate of a policy you have not deployed?** With real data you can never know. Here you can.

## The world

Each transaction gets one of four actions: route A or B, with or without 3DS.

| Effect | Model |
|---|---|
| 3DS friction | 5% to 21% abandonment, higher on mobile, new shoppers and small baskets |
| 3DS approval lift | Issuers approve authenticated transactions more often |
| 3DS against fraud | 85% of fraudsters fail authentication; liability shifts to the issuer |
| Routing | Route B costs more per transaction but suits some issuers better |
| Fraud | Depends on country risk, new shopper, amount, velocity and time of day |
| Chargebacks | Arrive 15 to 120 days after the transaction |

Logged traffic comes from a typical rule policy (3DS on amounts over €150 or risky countries, always route A) with 10% uniform exploration and logged propensities.

## Results

`python run_eval.py`: 112 s on one CPU core. Policies are learned from 80,000 logged transactions. True values are exact expectations over 200,000 fresh transactions. Estimators are graded on 10 independent logs of 40,000 transactions each, with 2-fold cross-fitted reward models.

### True value of each policy

| Policy | Net € per transaction | Conversion | Merchant-liable fraud | 3DS rate |
|---|---|---|---|---|
| Rules (production) | 13.190 | 80.1% | 59.2 bps | 26% |
| Never 3DS | 12.512 | 81.2% | 192.1 bps | 0% |
| Always 3DS | 13.303 | 76.3% | 0.0 bps | 100% |
| Model, unconstrained | 13.615 | **84.9%** | 123.8 bps | 28% |
| **Model, fraud-constrained (budget 12 bps predicted)** | **13.848** | 81.3% | 33.5 bps | 75% |
| Oracle (knows the true probabilities) | 14.134 | 84.7% | 54.2 bps | 52% |

The constrained policy earns **5.0% more per transaction than the rules, with 43% less merchant-liable fraud.** It also beats the unconstrained model. The unconstrained model maximises its own *predicted* revenue, keeps 3DS on only 28% of traffic to protect conversion, and pays for it in fraud that its model underestimates. The constraint acts as a guardrail on model error, not just a compliance cost. (The realised fraud rate, 33.5 bps, is above the 12 bps the model predicted: the model is optimistic about fraud, which is exactly the failure the guardrail limits.)

### Would the offline estimate have told you that?

| Policy | Effective sample | IPS bias | SNIPS bias | DM bias | DR bias | DR RMSE | DR 95% CI coverage |
|---|---|---|---|---|---|---|---|
| Rules (production) | 92.4% | −0.016 | −0.004 | +0.085 | +0.004 | 0.071 | 100% |
| Always 3DS | 3.4% | +0.050 | −0.075 | −0.130 | +0.024 | 0.221 | 80% |
| Never 3DS | 9.1% | +0.607 | +0.616 | +0.606 | +0.456 | 0.680 | 80% |
| Model, unconstrained | 2.9% | +0.314 | +0.155 | +0.035 | −0.093 | 0.645 | 90% |
| Model, constrained | 2.8% | +0.209 | −0.035 | −0.217 | −0.187 | 0.403 | 100% |

(Bias and RMSE in euro per transaction. Effective sample is the share of logged transactions that effectively count, from the importance weights.)

| Estimator | Kendall τ with the true ranking | Picked the truly best policy |
|---|---|---|
| IPS | 0.42 | 40% |
| SNIPS | 0.24 | 20% |
| Direct method | 0.50 | 50% |
| Doubly robust | 0.50 | 20% |

**The gap between the two model policies is €0.23 per transaction; the estimator error is €0.4 to €0.6.** With 10% exploration spread over four actions, a policy that disagrees with production on most traffic keeps under 3% effective sample. Even with every label matured, no estimator reliably picks the right policy, and the doubly robust estimate prefers the worse, unconstrained one in 60% of logs. This is the concrete version of *why the globally optimal payment policy can look worse offline but be better online*.

### Delayed fraud labels

The same logs, re-estimated with only the chargebacks known by each day. The shift is the DR estimate on that day minus the fully matured estimate, averaged over 10 logs.

| Evaluation day | Fraud labels known | Shift, unconstrained | Shift, constrained | Shift, rules | Logs where unconstrained looks better |
|---|---|---|---|---|---|
| 45 | 14% | +1.015 | +0.368 | +0.434 | **90%** |
| 90 | 57% | +0.474 | +0.060 | +0.223 | 80% |
| 150 | 100% | 0 | 0 | 0 | 60% |

Unseen fraud looks like a good sale, so early evaluation inflates every policy. It inflates most the policy that takes the most fraud risk: the unconstrained one gains €1.02 per transaction of phantom value at day 45. **Evaluate six weeks after logging and you ship the wrong policy in 9 of 10 cases.**

### Serving

Scoring one transaction (four actions, three gradient-boosted models) and applying the constraint takes 2.4 ms at p50 and 2.9 ms at p99 in Python on one core.

## What a team would do with this

1. **Buy exploration where it matters.** Uniform 10% exploration wastes most of its budget on actions no candidate policy takes. Targeted exploration on the traffic where candidates disagree would raise the effective sample for the decision actually being made.
2. **Never evaluate on immature labels without correcting for them.** Either wait for the chargeback window or model the chargeback arrival hazard. Naive early numbers are biased in a predictable direction: toward risk.
3. **Treat offline evaluation as a filter, not a verdict.** It reliably separates bad policies (never 3DS) from plausible ones. The final choice between close candidates needs a guarded online test.
4. **Constrain policies on the quantity the model is worst at.** Here that is fraud. The constraint recovered value the unconstrained model lost to its own optimism.

## Run it

```bash
pip install -e ".[dev]"
python -m pytest -q          # 8 tests: exact value vs Monte Carlo, sampling, liability shift, on-policy IPS,
                             #          DR with a perfect model, hidden fraud before chargebacks, λ tuning
python run_eval.py           # about 2 minutes; python run_eval.py --quick for CI
open results/report.html
```

## Limitations

- The outcome model is a designed simulator. Real issuer behaviour, fraud adaptation to 3DS and scheme rules are far richer; the lab tests estimation methodology, not payment economics.
- Routing is two options and authentication is on or off. There are no retries, exemptions or network tokens.
- Fraudsters do not adapt to the deployed policy, so the lab cannot show the policy feedback loop that real fraud systems face.
- Reward models are gradient-boosted trees without calibration. Better calibration would reduce DM bias, but not the effective-sample limit.

## Next steps

1. Targeted exploration (explore only where candidate policies disagree) and its effect on estimator error per exploration euro.
2. A chargeback-hazard correction for early evaluation, graded against the day-150 truth.
3. Add retries and a fraud-graph feature, keeping exact ground truth.
4. A guarded online test simulator: sequential testing with a fraud stop-loss.
