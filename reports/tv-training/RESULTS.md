# TV RoBERTa pilot results — October 6, 2026

Completed exactly one epoch per arm on the existing Colab Tesla T4. Both adapted the same pinned spoiler checkpoint using identical 744 training / 198 validation sentences and seed 42. Four training shows and three validation shows are disjoint. This is exploratory agreement with original TV Tropes markup labels, not product gold. No test evaluation or threshold fitting. No second epoch launched.

At fixed cutoff 0.5:

| Model/input | Spoilers caught (of 168) | Safe sentences flagged (of 30) | Precision | Recall | Spoiler F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original checkpoint, sentence | 19 | 8 | 70.4% | 11.3% | 19.5% | 42.3% | 76.4% |
| After TV adaptation, sentence | 144 | 19 | 88.3% | 85.7% | 87.0% | 61.2% | 92.8% |
| After TV adaptation, context | 168 | 30 | 84.8% | 100% | 91.8% | 50.0% | 83.1% |
| Always flag spoiler | 168 | 30 | 84.8% | 100% | 91.8% | 50.0% | 84.8% |

The context arm predicts spoiler for every validation sentence at cutoff 0.5; its high F1 is entirely reproduced by the trivial always-spoiler rule. Scores range 0.535–0.579. This does not establish that context can never help, but this one-epoch setup is not a useful contextual classifier at the fixed cutoff.

Sentence-only adaptation improves ranking and balanced accuracy over the original checkpoint on these TV examples, but flags 63.3% of safe sentences. This is not good enough to deploy. There are only 30 safe validation examples and substantial source-label/context quality concerns; large apparent F1 is not evidence of broad movie/book/transcript generalization.

Actual training time: sentence 27.0 seconds, context 29.7 seconds. Each arm took about 35 seconds including final evaluation/model saving; setup and ZIP packaging brought the overall launcher to roughly 2.5 minutes. Downloads take additional time. See comparison.json, per-arm reports, and prediction caches for exact metrics.

Recommended next step: improve safe-example coverage and label/grounding review, create matched positive/negative sentence pairs sharing the same plot context, and expand show diversity before a larger contextual run. Retain the original baseline and this sentence-only adaptation as separate checkpoints. Any threshold selection must be on development data, with an independent test afterward. No automatic extra training.
