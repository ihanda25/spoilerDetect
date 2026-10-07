# Annotation guide — pending human review

This is an assistant-authored development starter pack, not a completed human-reviewed benchmark. It follows the proposed policy in `reports/product-evaluation/PLAN.md`. All 64 rows await human review; 52 have provisional binary assistant labels and 12 have no binary proposal. Reviewers should not see those proposals or predictions during their first pass.

## Review procedure

1. Open `REVIEW_SHEET.md` and `review-response-template.json`. The sheet has neutral, shuffled review IDs and full text, including all long reviews. The JSON template contains blank decisions only. Do not open `examples.jsonl`, `assistant-proposals.json`, `review-key.json`, `predictions.jsonl`, or `RESULTS.md` until submitting your independent pass: those expose proposed labels, challenge types, or model outputs.
2. Use the supplied fictional public premise and private story fact. Assume a new viewer who only knows the public premise. Label **spoiler** for a concrete later outcome, hidden identity, resolution, fate, or twist; **safe** for premise-only statements, craft opinions, or genuine unconfirmed speculation. A warning does not neutralize a reveal. Negation can disclose a fact. Keywords alone do not determine the decision.
3. Select **uncertain** when omitted conversation, pronouns, truth versus theory, premise boundaries, or insinuation prevent a confident decision. Do not force uncertainty into a safe label. A question can reveal a fact indirectly; document the policy choice. Record an exact triggering quote for spoilers, or a quote illustrating ambiguity for uncertain cases. Explain missing context; safe rows can leave the quote null.
4. Enter human_label (`spoiler`, `safe`, `uncertain`), rationale, reviewer_id, and ISO-8601 UTC reviewed_at. Change status to `human_reviewed` only after an actual person labels the row. Preserve their first independent pass in a separate response file; do not overwrite the assistant proposals.
5. Have a second person independently review uncertain rows and policy-sensitive disagreements when possible. Preserve both passes. An adjudicator records a final decision, explanation and identity; use `adjudicated` only after that step. Unresolved cases stay uncertain. A single human pass is not adjudication.
6. Only then use `review-key.json` to join review IDs to example IDs and inspect model predictions. If transferring final labels into example records, keep proposed_label unchanged and populate annotation fields, including triggering_quote. Human metric computation is deliberately not automated in this initial scorer: it reports assistant-label agreement only, even if annotation fields are later filled. Implement a separately named, audited human-label evaluation once real reviews exist.

## Provenance and schema

`annotation.schema.json` defines each canonical example record. `review-response.schema.json` defines the human response template. No human labels are prefilled. Every canonical row preserves author, source type, date, fictional work context, work grouping, and proposed rationale. All examples are English. Transcript rows additionally have a synthetic video ID; review pairs share a diagnostic pair ID. Assistant-proposed evidence quotes/offsets are kept separately in assistant-proposals.json when an explicit reveal can be matched. Evidence offsets are zero-based Python character indices, end-exclusive, into the full text, not the truncated input. They are proposals, not human span judgments.

There are no source URLs because content was authored here, not acquired from the web. Timestamps are invented and have no claim to real video provenance. Fictional work titles may coincidentally resemble existing titles; story facts here are invented. No external-content permissions were assessed or needed for copied material because no corpus text was copied. This is not a general legal determination.

Keep all rows for a fictional work together in any future partition, including contrastive review pairs. The pack is entirely development data and must never be promoted to a sealed test after inspecting predictions. Four works and templated contrasts are insufficient for work-disjoint generalization estimates. Exact duplicates: the context-free transcript probe is intentionally repeated four times, under four work contexts; these uncertain rows are excluded from binary metrics. Near-duplicate templates across works are deliberate, not independent observations.

## Coverage and gaps

There are 16 examples per surface: headlines, comments, full reviews, and timestamped YouTube transcript passages. Challenges include explicit reveals, public premises, craft opinions, speculation, warning keywords, sarcastic replies, indirect questions, missing antecedents, synthetic ASR-like text, and early/late review reveals. Long reviews have artificial repetitive padding to isolate truncation; they are not realistic samples of review writing.

Missing: real user/source distributions, genuine transcription errors, adjacent passage context, negated true facts, actual multilingual traffic, franchise holdouts, fact verification against real works, viewer-progress personalization, event-span/video false-alarm evaluation and visual spoilers. Do not interpret the provisional metrics as representative production accuracy, human agreement, or evidence that the architecture has reached its ceiling.

## Reproduction

From the repository root:

```sh
.venv/bin/python scripts/build_development_benchmark.py
.venv/bin/python scripts/prepare_benchmark_review.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/evaluate_development_benchmark.py
```

Validate saved artifacts without inference with `.venv/bin/python scripts/validate_development_benchmark.py`. This uses a dependency-free validator for the specific JSON Schema keywords present in this pack, plus scorer and artifact invariants.

The builder rewrites canonical synthetic examples; review preparation rewrites only the blank response template and generated sheet/key/proposals. Save filled responses under a distinct filename before regeneration. Inference loads the local review-only epoch-2 checkpoint, sets eval mode and inference mode, uses CPU with two threads and batch size four, and never accesses test rows or tunes the frozen cutoff 0.21579217910766602. No model weights are saved or updated. The base-loader warning about a newly initialized classification head occurs before strict loading of the saved trained state; the saved classifier replaces that temporary initialization.
