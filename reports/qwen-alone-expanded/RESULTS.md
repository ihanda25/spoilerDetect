# Expanded Qwen-alone results — October 6, 2026

Status: completed all 267 examples.

| Slice | Spoilers caught | Safe falsely flagged | Abstentions on binary labels | Pending |
| --- | ---: | ---: | ---: | ---: |
| real_content_dev | 5/5 | 1/138 | 19 | 0 |
| source_plot_challenge | 40/40 | 2/20 | 0 | 0 |
| tv_markup_dev | 18/30 | 8/30 | 10 | 0 |

Invalid answers: 27; API JSON-generation failures: 1. These become UNCERTAIN, not safe. Recall includes abstained positives as misses.

Real-content and source-plot challenge labels are AI-reviewed; TV labels are corpus markup. None are human gold. Only five real-content positives are present. TV is a balanced 30/30 diagnostic from three shows, not natural prevalence or a new unseen benchmark. Source-plot challenges share sources and may be familiar to pretraining. Keep slices separate; do not interpret aggregate accuracy as production performance.

Same Qwen3.8-27B model, frozen system instructions, title/public premise inputs, temperature zero and 512 output-token ceiling. No plot retrieval or training. 79 identical earlier answers are reused with provenance. Updated transport: 4s minimum spacing, 50% token headroom, rate-limit headers and bounded short 429 retries. Specific HTTP 400 json_validate_failed responses are recorded as invalid abstentions so unrelated examples continue. Other API errors still stop. No paid fallback or purchases.

### Accurate-context TV diagnostic and scoring correction

Completed 14 targeted Qwen examples with manually checked episode context. Both arms classified all seven source-audited spoilers correctly; context resolved one vague safe reference. Three context passage quotes contained a control character in place of an apostrophe and failed evidence validation. No spoiler recall improvement demonstrated. Labels are a non-blind AI audit, not human gold. See [diagnostic results](../qwen-tv-verified-context/RESULTS.md).

Separately, whitespace-only rescoring of the original 60 TV examples restores eight invalid answers, including five spoilers: original-markup recall is 23/30 (76.7%) rather than 18/30 (60%). Original outputs and historical tables are preserved; this is a validator correction, not a model improvement.
