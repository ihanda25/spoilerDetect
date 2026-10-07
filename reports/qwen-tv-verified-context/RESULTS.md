# Qwen accurate-context TV diagnostic

**Targeted development experiment, selected after inspecting errors. This tests accurate supplied context, not retrieval quality or generalization.**

No training or purchases. Same cached alone inputs and frozen classification prompt; context adds manually curated source-backed passages. Whitespace-only normalized quote validation is used for both arms.

| Source row | AI policy label (1 spoiler, 0 safe, None unresolved) | Alone | Context |
|---|---|---|---|
| 547 | 1 | SPOILER | SPOILER |
| 395 | 1 | SPOILER | SPOILER |
| 652 | 1 | SPOILER | SPOILER |
| 627 | 1 | SPOILER | UNCERTAIN |
| 328 | None | SAFE | SPOILER |
| 565 | 0 | UNCERTAIN | SAFE |
| 307 | 0 | SAFE | SAFE |
| 545 | 0 | SAFE | SAFE |
| 593 | 0 | SAFE | SAFE |
| 638 | 0 | SAFE | SAFE |
| 568 | 0 | SAFE | SAFE |
| 562 | 1 | SPOILER | UNCERTAIN |
| 324 | 1 | SPOILER | UNCERTAIN |
| 578 | 1 | SPOILER | SPOILER |

## Interpretation

Raw decisions: both arms identify all seven source-audited spoilers. Alone marks five of six safe references SAFE and abstains on the vague ending reference; context marks all six SAFE. The ambiguous financing example changes SAFE to SPOILER but remains unscored. There is no demonstrated spoiler recall gain here.

Three context responses fail exact evidence validation because U+2019 apostrophes in supplied passages become U+0019 control characters in returned passage quotes. Their raw labels are SPOILER; this is a quotation encoding failure, not a semantic miss. The validator remains strict on content. Accepted evidence-backed recall is consequently 4/7 versus 7/7 for alone. Raw-label agreement must not be presented as evidence-backed reliability.

Next: fix quote serialization (prefer returned character offsets or passage IDs with locally extracted quotes), then test automatically retrieved context on a larger, independently labeled set. This targeted set cannot establish overall model quality.

## Reviewed-label metrics

```json
{
  "alone": {
    "all": {
      "invalid": 0,
      "positives": 7,
      "tp": 7,
      "uncertain_reference": 1,
      "negatives": 6,
      "abstentions": 1,
      "tn": 5,
      "recall": 1.0,
      "false_positive_rate": 0.0,
      "coverage": 0.9230769230769231,
      "precision": 1.0
    },
    "tv_markup_dev": {
      "invalid": 0,
      "positives": 7,
      "tp": 7,
      "uncertain_reference": 1,
      "negatives": 6,
      "abstentions": 1,
      "tn": 5,
      "recall": 1.0,
      "false_positive_rate": 0.0,
      "coverage": 0.9230769230769231,
      "precision": 1.0
    }
  },
  "context": {
    "all": {
      "invalid": 3,
      "positives": 7,
      "tp": 4,
      "abstentions": 3,
      "uncertain_reference": 1,
      "negatives": 6,
      "tn": 6,
      "recall": 0.5714285714285714,
      "false_positive_rate": 0.0,
      "coverage": 0.7692307692307692,
      "precision": 1.0
    },
    "tv_markup_dev": {
      "invalid": 3,
      "positives": 7,
      "tp": 4,
      "abstentions": 3,
      "uncertain_reference": 1,
      "negatives": 6,
      "tn": 6,
      "recall": 0.5714285714285714,
      "false_positive_rate": 0.0,
      "coverage": 0.7692307692307692,
      "precision": 1.0
    }
  }
}
```

Original markup agreement is saved separately. Source-backed AI audit is not human gold; three original safe labels disclose later events, one original spoiler label is a vague ending reference, and one reference remains ambiguous.

The full 60-row baseline has five additional valid spoiler predictions after whitespace normalization: recall changes from 18/30 to 23/30. This is a scoring correction, not a new inference gain. Eight previously invalid quotes become valid; original caches remain intact.
