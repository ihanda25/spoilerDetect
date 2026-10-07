# Three-model text-only comparison

User authorized 2026-10-05; free Colab T4, no API key and no training. **User explicitly requires zero spending: do not purchase compute units, enable subscriptions, use paid APIs, or provision paid runtimes. Stop if free GPU access is exhausted.** New notebook:
https://colab.research.google.com/drive/1O3uMsIJqaXK0-5f27l9F2vc9EYmhVDEX

- MiniLM: owned review-only epoch-2 checkpoint, exported as safetensors. Threshold 0.21579217910766602 from prior IMDb validation.
- RoBERTa: Zritze/imdb-spoiler-robertaOrigDatasetLR1, revision 56fee120f8495ccfc5001e3fbd1478656d17001a, threshold 0.5.
- LLM: Qwen/Qwen2.5-3B-Instruct, Hub revision resolved and recorded before inference. FP16 GPU inference, greedy SAFE/SPOILER/UNCERTAIN output. No retrieved context, examples, work-title metadata or annotation rationale is supplied. Model has pretrained knowledge.

147 development excerpts are shared across all systems; 98 held-out test excerpts are not packaged. All models see only excerpt text. Classifiers truncate at 512 tokens; LLM allows up to 2048 including its fixed policy prompt and refuses longer inputs. Actual truncation counts are reported. This compares configured systems; classifier thresholds are not equally calibrated on this domain. No threshold fitting occurs.

21 human annotations (9 safe, 6 spoiler, 6 uncertain), and 126 AI annotations (112 safe, 2 spoiler, 12 uncertain) are reported separately. Human original timestamps could not be recovered, groups are not verified, and AI judgments are not gold. This is exploratory development agreement, not final product accuracy. Null reference labels are excluded; LLM abstentions and invalid responses are counted, with recall including abstentions and answered-only metrics. A tiny positive count and source concentration limit conclusions.

`input-manifest.json` records the ZIP hash and checkpoint epoch. Package lives at ignored `models/three-model-comparison-inputs.zip` (~123 MB). It contains owned model weights, blind dev JSONL, separate annotations, and `compare_three_models.py`; never publish this ZIP to Git. Builder: `scripts/build_three_model_colab.py`. Notebook: `notebooks/Spoiler_Three_Model_Comparison.ipynb`.

Predictions save atomically per row under `/content/three-model-comparison/results`. Re-running a cell resumes matching completed predictions in the same runtime. Disconnecting can destroy `/content`; archive downloads are requested after each model. Downloads must be saved before runtime expiration. `comparison.json` and each model cache contain configuration fingerprints and timing; comparison JSON contains no raw excerpt text.

Current state: **STOPPED at user request to avoid spending.** Resource panel revealed an existing Colab Pro subscription, 398.99 available compute units, rate ~2.14 units/hour across two active sessions. Earlier statements that this particular runtime was free were incorrect/unverified. No subscription, compute-unit purchase, paid API, or paid cloud resource was activated by this task; existing units may have been consumed and exact consumption is unknown. Inference interrupted; user explicitly approved Disconnect and delete this runtime. Deletion confirmed in UI: Reconnect T4 / Not connected to runtime. A second session was not modified. Do not restart any hosted GPU without verifying it avoids paid-unit consumption.

MiniLM and RoBERTa completed 147 predictions each. Partial results transcribed from visible notebook output are in `partial-results.json`: both classified all 15 binary human-reviewed development examples safe (0/6 labeled spoilers caught). On separate AI labels, MiniLM caught 0/2; RoBERTa caught 2/2 with 2 false positives. These are tiny, unverified diagnostic sets, not production metrics. LLM did not finish. No fair three-way result exists.

Notebook output preserves the partial summary and an embedded Download preserved comparison results link. Automated ZIP download timed out; local receipt of the full prediction archive is unverified. Runtime deletion removed temporary model downloads and caches; local original checkpoint/input ZIP remain safe. Local notebook keeps the full runner for a future run; Colab last cell currently exports partial results. Verify zero-cost execution first.

## User hold (2026-10-05)

User explicitly requested: hold off on the LLM test; do it later. Do not launch, reconnect a runtime for, download weights for, or resume the LLM baseline until the user asks to resume. MiniLM/RoBERTa partial results are preserved. Zero-spending constraint remains in force.

## Resume authorized (2026-10-05)

User revoked the LLM hold and explicitly chose **use existing Colab compute units, no purchases** after being informed that Pro runtimes use existing units. This supersedes the zero-unit restriction for this comparison only; no subscriptions, compute-unit purchases, paid APIs, or other paid resources authorized. Plan: recover classifier caches from the notebook’s embedded result ZIP, upload lightweight LLM-only inputs, run Qwen on the same blind147 dev rows, then preserve results.

## Completed results (latest state, 2026-10-05)

All three models now have predictions for all 147 shared dev excerpts. Recovered MiniLM147/RoBERTa147/Qwen92 caches from the notebook’s persisted ZIP; resumed the remaining55 Qwen examples on an authorized existing-unit T4 session. The resumed cell took ~119 seconds including model download/loading. Qwen revision aa8e72537993ba99e69dfaafa59ed015b17504d1. All predictions passed exact dataset/text-hash checks; zero invalid outputs and zero truncated excerpts. No thresholds or prompts changed on resume.

Final local `results.json` preserves overall and surface metrics, runtime metadata, model revisions, policy prompt and limitations. Full original report, Qwen cache and complete archive remain embedded in the saved Colab output as download links; automatic local receipt of the raw archive is not verified. Do not present the earlier `partial-results.json` as current results.

On the15 definite human dev labels (6 positive/9 negative), all systems caught0/6; classifiers answered15 and Qwen answered6, abstaining9. Several human positives appear to be production/review opinions rather than later story revelations under the frozen rubric; preserve original decisions and clarify policy before adjudication. No human labels were overwritten.

On114 definite AI dev labels (2 positive/112 negative): MiniLM TP0/FP0/FN2; RoBERTa TP2/FP2/FN0; Qwen TP1/FP0/FN0 plus83 abstentions, including1 positive. Qwen covers31/114; its100% answered-only accuracy excludes83 abstentions and is NOT a valid blanket quality claim. Across all147 excerpts Qwen labels39 safe,2 spoiler,106 uncertain (72.1% uncertain). This weak LLM-alone baseline does not establish how a stronger LLM or RAG would perform.

Existing compute units were explicitly authorized, no purchases or API requests. Shutdown confirmation was requested after preserving results; current runtime remains allocated pending the user’s answer. A separate pre-existing session was not modified.
