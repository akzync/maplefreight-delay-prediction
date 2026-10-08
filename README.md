# MapleFreight — Delivery Delay Prediction & Slack Alerts
AML 3303 · Assessment 1

## Problem
MapleFreight moves ~8,000 shipments a week and only finds out a delivery is late when the customer calls. On-time performance fell from 94% to 88%. This project predicts, at dispatch time, which in-transit shipments will be late and posts a Slack alert to `#dispatch-alerts` while there is still time to re-route or warn the customer.

## Dataset
`maplefreight_delivery_delay_dataset.csv` — 6,035 shipments × 23 columns; target `delivered_late` (20.6% late).
Cleaning: 32 exact duplicates + 3 conflicting repeated `shipment_id`s removed (→ 6,000 rows); `service_level` case normalised; 40 weights recorded in grams converted to kg; missing weather → `Unknown`; other gaps median-imputed inside the model pipeline. `actual_transit_hours` is **excluded** (only known after delivery = leakage).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Slack webhook URL into .env
jupyter notebook MapleFreight_Delivery_Delay_Prediction_c0969478.ipynb   # Kernel > Restart & Run All
```
Without a webhook the notebook still runs and prints a preview of the message instead of sending it.

### Slack setup
1. Create a Slack workspace and a channel `#dispatch-alerts`.
2. Create a Slack app → enable **Incoming Webhooks** → *Add New Webhook to Workspace* → pick `#dispatch-alerts`.
3. Put the URL in `.env` as `SLACK_WEBHOOK_URL=...` (never commit `.env`).
4. Run the last section of the notebook, then save a screenshot of the message as `screenshots/slack_alert.png`.

## Key findings
- **Weather** is the biggest driver: late rate 14% in clear weather, 34% in snow, 46% in storm.
- **Carrier**: contract owner-operators are late 36% of the time vs 13% for the own fleet; owner-operators exceed 55% in snow/storm.
- **More stops** (12% → 29% from 0 to 5 stops), **high traffic** (29.8% top quartile vs 13.1%), **late pickup** (>30 min: ~26% vs ~15%) and **Economy service** (26.6% vs 15.1% Express) all raise risk.
- Customer priority tier, origin/destination city and goods type carry almost no signal.

## Models and results (held-out test set, 1,200 shipments)
Imbalance handled with `class_weight="balanced"` and evaluation by precision/recall/F1/PR-AUC (not accuracy).

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Dummy (always "on time") | 0.00 | 0.00 | 0.00 | 0.206 |
| **Baseline**: logistic regression, raw features, threshold 0.5 | 0.44 | 0.75 | 0.555 | 0.601 |
| **Final**: tuned logistic regression + engineered features, threshold 0.48 | 0.44 | 0.78 | 0.56 | 0.599 |

Confusion matrices are in the notebook. The final model catches 193 of 247 late shipments with 249 false alarms.
Random Forest and gradient boosting were also tried and **did not beat the logistic regression** (CV PR-AUC 0.50 / 0.54 vs 0.556), and the engineered features added essentially nothing. The final model is only marginally better than the baseline; the notebook says so rather than overstating it.

## Chosen threshold: 0.48 — and why
Picked on the **validation** set by minimising expected cost, assuming a missed late shipment costs CAD 150 (service credit + re-delivery + account-manager time) and a false alarm costs CAD 40 (dispatcher check / customer pre-warning). **These costs are my assumptions**, not MapleFreight data; the notebook includes a sensitivity table (threshold ranges from ~0.28 to ~0.72 depending on the cost ratio). At 0.48, validation recall is ~72% and ~37% of shipments are flagged. On the test set this gives an expected cost of about CAD 18k vs CAD 37k with no model. Because ~37% alert volume is too many to review one-by-one, I recommend tiering alerts by score. Scores are class-weighted rankings, not calibrated probabilities, so the Slack message shows a "risk score".

## Recommendations (summary — full list in the notebook)
Treat snow/storm at dispatch as a trigger for buffers and customer pre-warnings; keep severe-weather loads with the own fleet and add SLAs for owner-operators; limit stops on tight windows; re-score and escalate at late pickup; loosen Economy promise windows; tier Slack alerts; replace the assumed costs with real ones and retrain monthly.


- **Tool used:** Claude (Anthropic) — data inspection, notebook code, and drafting written findings.
- **One case where the output was wrong or needed correcting:** (1) a first version of the notebook picked a tuned gradient-boosting model as the final model by assumption; cross-validation showed a logistic regression was actually stronger, so the model selection was changed to be data-driven and the baseline definition was changed to raw-feature logistic regression. (2) The first Slack message showed the score as a percentage ("97%"), which wrongly implies a calibrated probability since `class_weight="balanced"` inflates scores; it was changed to "risk score 0.97".
- **How I verified:** I re-ran the notebook from a clean kernel, checked the quoted figures in the written findings against the printed outputs, and confirmed no webhook URL appears in the repo or outputs.

## Repository contents
`MapleFreight_Delivery_Delay_Prediction_c0969478.ipynb` · `README.md` · `screenshots` · `requirements.txt` · `.gitignore` · `maplefreight_delivery_delay_dataset.csv`
