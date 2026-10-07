# Episode context preparation — October 6, 2026

Collected 416 episode plots across seven pilot series, with 27 detailed episode articles and 795 searchable chunks. Source URLs, retrieval dates and cached HTML hashes are recorded in episode-source-manifest.json. Public source text remains in ignored local data; source usage caveats from the earlier audit still apply.

The frozen pilot remains 200 training and 60 validation examples, with identical labels and IDs. Episode passages attach to 258 of 260 examples: 50 use explicit episode-title matching; the other 210 use within-show lexical search (including two without retrieved passages). Attachment is not verified event coverage. Query/scoring use the sentence and work identity, never the spoiler label. Retrieved passages retain source URLs; series background has separate attribution.

Inputs include up to 96 tokens of series background and two retrieved episode passages. All pilot pairs fit RoBERTa's 512-token limit: maximum 496 tokens, no pilot sentence or passage truncation. Train/validation works and episode source URLs remain disjoint. All seasons are included, with no viewer-progress personalization. This is a development preparation pack, not a representative test set.

Six focused assistant spot checks found one supported proposition, four partial propositions, and one retrieval miss. See episode-quality-judgments.json. These checks are deliberately limited and do not estimate corpus accuracy. Original fragments, ambiguous work references and markup labels still need review. All rows retain training_allowed=false and human_verified=false. No model inference or training started.

## Reproduction

Install requirements-training.txt and requirements-context-data.txt in the project environment. The pinned tokenizer assets are documented in tokenizer-manifest.json; no model weights were downloaded for this step.

```sh
.venv/bin/python scripts/test_episode_context.py
.venv/bin/python scripts/collect_episode_context.py --offline
.venv/bin/python scripts/add_episode_context.py
```

Omit --offline only when intentionally fetching missing public sources. Cache and episode corpus live under data/raw/episode-context. Output pack: data/processed/sentence-context-pilot-episodes. Five regression checks cover title matching, table parsing, multipart/nested plot sections and within-work retrieval. Output assertions check frozen IDs/labels, token budget and split separation.

Next: expand label and grounding review, improve failed retrieval, then decide whether the paired pilot justifies a bounded hosted-GPU experiment. Do not automatically train.

October 6: user authorized a bounded one-epoch hosted-GPU TV adaptation comparison using 744 train / 198 validation mapped candidates. See reports/tv-training/PLAN.md (relative to project root). Original preparation quality flags remain unchanged; this is exploratory corpus-label training, not reviewed gold.

October 6 experiment completed: one epoch each on Colab T4. Sentence-only catches 144/168 spoilers and flags 19/30 safe examples; context catches 168/168 but flags all 30 safe examples at cutoff .5. See reports/tv-training/RESULTS.md and comparison.json. No automatic second epoch.
