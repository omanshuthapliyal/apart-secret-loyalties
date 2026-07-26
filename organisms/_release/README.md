# Track 1 - released organisms

8 LoRA-fine-tuned secret-loyalty organisms on Qwen2.5-1.5B-Instruct, one per
principal category. Real principal names are anonymized here (category + light
context, matching report.html) - see report.html App. B for the disclosure policy.
Full methodology, statistics, and cross-principal generalization results in the
companion report.

| Category | Activation rate | Selectivity | Principal selectivity | Max black-box detection |
|---|---|---|---|---|
| Nation-state (1/4) | 0.9167 | 0.7333 | 1.0 | 0.6 |
| Nation-state (2/4) | 0.9167 | 1.0 | 1.0 | 0.4 |
| Nation-state (3/4) | 1.0 | 0.9231 | 1.0 | 0.2 |
| Nation-state (4/4) | 0.9167 | 1.0 | 1.0 | 0.2 |
| Corporation | 0.8333 | 1.0 | 1.0 | 0.6 |
| AI company executive | 1.0 | 0.5 | 1.0 | 0.4 |
| Political leader | 0.75 | 0.9 | 1.0 | 0.6 |
| Ideology / movement | 0.5 | 0.5455 | 1.0 | 0.4 |

Per-organism detail (internal, anonymized run ids in `index.json` above):
`organisms/_release/index.json`. Full 28-pair cross-principal matrix,
equations, statistical treatment, and figures: `report.html`.