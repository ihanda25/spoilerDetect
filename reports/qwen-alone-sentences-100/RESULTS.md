# Qwen-only: 100 fresh TV sentences

Complete: True. No retrieved plot context or training. Title and public premise supplied.

| Show | Spoilers caught | Safe falsely flagged | Abstentions |
|---|---:|---:|---:|
| all | 35/50 | 14/50 | 4 |
| Firefly | 9/13 | 2/12 | 1 |
| Fringe | 10/12 | 5/13 | 0 |
| Sherlock | 8/13 | 1/12 | 0 |
| PersonOfInterest | 8/12 | 6/13 | 3 |

Recall includes abstained positives as missed spoilers. Whitespace-only normalized quotation validation; other evidence edits remain invalid. Corpus markup is not human gold and may disagree with the spoiler policy. This balanced sample does not represent natural prevalence. Rows come from the previous classifier training partition, but none were included in the previous Qwen sentence evaluation. No held-out or production-accuracy claim.

Errors and abstentions are saved in errors.json with source labels, text, raw model decisions, and validation outcomes. Review them before interpreting disagreements as model failures.

## Follow-up error review

The 31 disagreements/abstentions were reviewed separately: 13 dataset-policy disagreements, 5 model classification errors, 2 quote-validation failures, and 11 unresolved excerpts or policy boundaries. This non-blind AI review does not revise overall accuracy; the other 69 examples remain unaudited. [Full review](ERROR_REVIEW.md). Original labels and scores are preserved.

## Full prediction-hidden AI review

Two reviewers independently labeled disjoint sets of 50 anonymized inputs without Qwen predictions or original source labels. Fixed policy; all labels locked before scoring. References: 38 spoiler, 45 safe, 17 unresolved. Qwen caught 30/38 spoilers (78.9% recall), falsely flagged 8/45 safe sentences (17.8%), and matched 67/83 resolved references (80.7%), including abstentions as non-correct. Precision 78.9%. One abstention on resolved references. The 17 unresolved references are excluded from binary scores. No plot-source verification was used in this review; these are AI policy judgments, not human gold. Original scores remain preserved. [Full review](blind-review/RESULTS.md).
