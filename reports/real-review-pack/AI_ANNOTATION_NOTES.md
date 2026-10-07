# AI annotation notes

This sidecar contains AI first-pass labels for candidate rows 32–245 (214 records). Labels apply only to the displayed excerpt, read with its work title/context: `0` means no revealing narrative event, outcome, twist, or character fate; `1` means such a reveal is present; `null` means plot context or spoiler status is unclear. Premise descriptions, opinions, and production/news items are generally safe. No source lookup was used.

These records are AI-reviewed only. This does not assert human verification or evaluation eligibility. Rows 1–31 are outside this sidecar batch.

Final semantic review: 182 safe, 3 spoiler, 29 uncertain, with distinct short rationales for each of the 214 records. An initial blanket-default draft was rejected by the parent and replaced after individual review. No model scores were consulted.

Combined file: `annotated-human-and-ai.jsonl` preserves 31 human and 214 AI decisions. The candidate text, original split, and group IDs are unchanged. All grouping checks remain pending and evaluation eligibility remains false.

Parent spot-check corrected `real-4b448a2503ec4385`: praising a finale without disclosing events is safe. The prior AI decision is retained in correction metadata. Final AI totals: **183 safe, 2 spoiler, 29 uncertain**. This was a limited spot-check, not exhaustive independent adjudication.
