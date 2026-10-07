# Context comparison results — October 5, 2026

Completed on Colab Tesla T4 in 164.194 seconds including setup/model loading. No training, purchases or API calls. All three arms produced 207 valid cached outputs; no truncation or invalid LLM outputs. Local fingerprint checks and report recomputation passed.

| Slice | Model | Spoilers caught | False alarms | Binary rows answered |
|---|---|---:|---:|---:|
| Real content | RoBERTa | 3/5 | 2 | 143/143 |
| Real content | Qwen title only | 2/5 | 2 | 59/143 |
| Real content | Qwen title + context | 1/5 | 2 | 52/143 |
| Plot challenge | RoBERTa | 25/40 | 3 | 60/60 |
| Plot challenge | Qwen title only | 32/40 | 0 | 35/60 |
| Plot challenge | Qwen title + context | 34/40 | 14 | 48/60 |

Recall counts abstained positives as misses. Four uncertain AI reference labels are excluded from binary metrics. The real-content slice has only five positives; headline/review subsets have none, and the ten transcript fragments come from one subtitle track. These are AI development judgments, not human gold or production accuracy.

On the common context-covered real-content subset (34 examples, 32 binary labels), RoBERTa answered 32/32 and caught 3/5; title-only Qwen answered 12/32 and caught 2/5; context Qwen answered 5/32 and caught 1/5. All three had zero false alarms on this subset. The two all-real-content false alarms in the context arm are unchanged title-only fallbacks.

Context increased challenge recall from 32/40 to 34/40 while introducing 14 false alarms among 20 safe examples (70%). It supplied no correct SAFE decisions on this slice: the other six safe rows were abstentions. This setup therefore does not support switching to the context model. Source-derived plot fragments and context share Wikipedia sources; familiar works and within-work dependence limit generalization even if performance were strong.

There were 27 paired decision changes: 7 in real content (one previously detected spoiler and six safe decisions became abstentions) and 20 in the challenge. See `paired-changes.json` for exact IDs and fixed reference labels. Context had no invalid outputs; the failure is in decisions/abstentions rather than parsing.

Measured GPU inference excluding load/setup: RoBERTa 2.27 seconds for 207, title-only Qwen 42.54 seconds for 207, context Qwen 28.25 seconds for the 94 supplied-context examples. Context reuse of 113 title predictions costs zero additional inference but requires their underlying 21.72 seconds in a standalone fallback deployment; do not interpret zero cached latency as free inference. On the common 94 examples, mean latency was 0.221 seconds for title-only versus 0.301 seconds with context.

Keep fixed RoBERTa as baseline. Next inspect the context false alarms and abstentions, then freeze a revised task formulation on a separate development batch with paraphrased/full natural excerpts and stronger human review. Do not tune against the held-out 98 examples or fine-tune from these results. A later larger LLM or revised structured prompt is an experiment, not an assured fix.

Saved artifacts: comparison.json, roberta.json, llm_title.json, llm_context.json, spoiler-context-results.zip, verification.json. Runtime shutdown approval is pending; the runtime may continue to use existing units while left allocated.
