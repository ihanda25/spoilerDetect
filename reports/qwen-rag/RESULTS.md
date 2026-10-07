# Qwen3.8-27B partial results — October 6, 2026

Groq Free-plan experiment stopped on a reproducible HTTP 400 `json_validate_failed` response. The identical diagnostic retry failed the same way. 79 alone answers and 78 RAG answers are saved; use the 78 common IDs for paired aggregate comparisons. There are 94 planned inputs, so this is not a completed evaluation. No fallback, prompt change, additional training or purchases occurred.

The real-content development slice completed in both arms: 34 examples, including 5 AI-labeled spoilers, 27 safe and 2 uncertain reference labels.

| Arm | Spoilers caught | Safe falsely flagged | Safe correctly classified | Abstentions on binary labels |
| --- | ---: | ---: | ---: | ---: |
| Qwen alone + title/public premise | 5/5 | 0/27 | 26/27 | 1/32 |
| Qwen + retrieved evidence | 3/5 | 0/27 | 26/27 | 1/32 |

The partial source-plot challenge paired slice has 29 spoilers and 15 safe examples. Qwen alone catches 29/29, with 1/15 false alarms and no abstentions; RAG catches 18/29, with no false alarms and 11/44 abstentions. The source-plot challenges share narrative sources with the supplied context and are not independent evidence of generalization.

Across the 78 common examples (76 binary references and 2 uncertain): alone catches 34/34 positives, flags 1/42 safe, and abstains on 1/76 binary examples. RAG catches 21/34 positives, flags 0/42 safe, and abstains on 12/76 binary examples. comparison.json additionally includes the unpaired 79th alone response, so its all-example denominators differ; the complete real-content slice is identical.

All 12 RAG abstentions are validator failures: 11 supporting quotes were not exact substrings of the cited retrieved passage, and one excerpt quote was not exact. The alone arm has one excerpt-quote failure. These are conservative uncertainty conversions rather than all being explicit model uncertainty. A textual mismatch can be paraphrase or invented evidence; manually inspect before interpreting it as hallucination. Quote presence alone does not establish entailment.

Interpretation: the larger Qwen model is a promising candidate for a stronger evaluation, but five real-content positives and AI-reviewed development labels cannot establish production accuracy. This retrieval/evidence pipeline underperforms the paired no-retrieval baseline. Short assistant-written summaries, lexical passage retrieval, exact-quote generation, and familiar story knowledge are confounded. Do not infer that RAG in general cannot help. Prioritize a new human-reviewed evaluation and accurate context coverage, and separately diagnose retrieval misses versus citation-format failures before changing models or scaling up.

Artifacts: alone.json, rag.json, inputs.json, manifest.json, comparison.json, pacing.json, status.json, last-api-error.json. Secret values and authentication headers were not exported. The eight earlier smoke responses are kept separately under pilot-smoke and excluded from these counts.
