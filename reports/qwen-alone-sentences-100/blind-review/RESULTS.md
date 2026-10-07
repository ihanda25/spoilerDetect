# Prediction-hidden review of all 100 Qwen sentences

Independent prediction-hidden AI labels, not human gold. Unresolved references excluded from binary scores; abstained positives count as missed spoilers. Full 100-row review locked before scorer read predictions.

Two independent AI reviewers each reviewed 50 anonymized examples. They received titles, excerpts, supplied public premises and a fixed policy, with no model answers or source labels. They were allowed to check cached episode plots, but neither used plot sources; these are excerpt-and-policy judgments, not source-verified facts. Labels were validated and locked before the scorer opened predictions. No second reviewer per sentence; no inter-rater agreement claim.

References: {'unresolved': 17, 'spoiler': 38, 'safe': 45}. Resolved source-label changes: 21.

| Show | Spoilers caught | Safe falsely flagged | Abstentions on resolved labels | Unresolved references |
|---|---:|---:|---:|---:|
| all | 30/38 | 8/45 | 1 | 17 |
| Firefly | 9/11 | 1/11 | 0 | 3 |
| Fringe | 11/13 | 2/10 | 0 | 2 |
| PersonOfInterest | 7/11 | 1/6 | 1 | 8 |
| Sherlock | 3/3 | 4/18 | 0 | 4 |

## Overall

```json
{
  "negatives": 45,
  "tn": 37,
  "fp": 8,
  "positives": 38,
  "tp": 30,
  "fn": 7,
  "unresolved_reference": 17,
  "invalid": 1,
  "abstentions": 1,
  "recall": 0.7894736842105263,
  "false_positive_rate": 0.17777777777777778,
  "precision": 0.7894736842105263,
  "coverage": 0.9879518072289156,
  "accuracy_with_abstentions_in_denominator": 0.8072289156626506
}
```

Accuracy includes abstentions as non-correct decisions but excludes unresolved reference labels. Original scores are preserved. An AI label review is not a human evaluation; results apply to this deliberately sampled development set only.

See POLICY.txt, locked-labels.json and LOCK.json for exact labeling decisions and integrity hashes. remaining-cases.json includes unresolved references, model mismatches and abstentions for follow-up.
