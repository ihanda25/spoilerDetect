# Product evaluation plan — 2026-10-02

## Authorization and scope

User wants spoiler protection across headlines, comments, full reviews, and YouTube transcripts. Current task: begin development benchmarking and research existing trained models. Local bounded inference and preparation are authorized. Do not run heavy training/fine-tuning; first tell the user so they can arrange a hosted GPU. No external GPU spend or paid inference is authorized. The earlier Apple MPS runs were GPU-assisted, not CPU-only.

## Proposed annotation policy (requires human review)

Assume the reader has not started the work, but may know its public premise. Mark a concrete later plot outcome, hidden identity, resolution, character fate, or twist as spoiler. Public premise, acting/style opinions, production details and genuine unconfirmed speculation are normally safe. Mere words like 'dies' or 'ending' do not determine a label. Negation can itself reveal a plot fact. A spoiler warning does not erase the spoiler that follows it.

Use three annotation decisions: spoiler, safe, uncertain. Uncertain includes missing context, unclear truth versus theory, viewer-progress dependence, and disputed premise boundaries. Do not force ambiguous examples into binary gold labels. Preserve the triggering text span, work ID, context, reasoning, label provenance, reviewer and review status. Keep any assistant-proposed labels separate from human labels.

This is a proposed evaluation policy, not an established user-approved definition or a claim that an unfamiliar plot event can always be verified from text. Model confidence is not a calibrated probability of truth or spoilerness.

## Evidence levels

1. Assistant-authored fictional diagnostic examples: demonstrate predictable failure cases, not representative accuracy. Human review pending.
2. Corpus development examples: preserve original labels and provenance; disclose label noise and overlap risks.
3. Human-reviewed real content across the four surfaces: needed before product-quality claims. Human reviewers should label before seeing model predictions; disagreements need adjudication. Do not call model-generated labels human-reviewed.
4. Locked held-out product benchmark: freeze works and source documents after development; evaluate final chosen system once. Keep it separate from model-selection examples.

A development starter pack is not a finished production benchmark. No outcome from a small balanced pack establishes deployment precision.

## Evaluation design

Group by work/franchise where known, and by source document/thread/video. Every segment from a review or video must stay in the same split. Check duplicates and near-duplicates. Preserve URL/source ID, acquisition date, usage permissions, language, surface, work identity and viewer-progress assumptions. For public pretrained spoiler classifiers, unknown IMDb overlap means our IMDb test split may not be independent of their training; use independently collected content for a fair deployment comparison.

Score the current review-only epoch-2 checkpoint at the frozen IMDb-validation cutoff 0.21579217910766602. That cutoff is a reference operating point, not a universal cutoff for new domains. Report per-surface precision/recall/F1 and false-positive counts, with denominators. Do not optimize on diagnostic challenge examples or the already-used IMDb test set. Fit any new cutoff on a separate labeled development slice and disclose it.

Record exact checkpoint/revision, tokenizer, label mapping, maximum length, truncation side, input construction, token counts, duration and device. Count uncertain cases separately. For thresholded scores, distinguish ranking improvement from moving along a precision/recall tradeoff.

## Transcripts

Evaluate sentence or short passage predictions with nearby text, keeping source timestamps. Preserve punctuation/ASR variants and ambiguous pronouns. Return time ranges, not only a whole-video label. Long videos require document-level false-alarm measurements: false warnings/hour, false-positive videos, and annotated spoiler-event recall/span coverage. Artificial transcript snippets do not establish these rates. Audio transcription quality, thumbnails, images and on-screen visual spoilers are outside a text-only benchmark.

## Decision after current work

Use the research shortlist to test an existing pretrained classifier before any new heavy training. A controlled same-input comparison should distinguish: changing models, adding context, changing labels, and changing truncation. If a GPU pilot becomes justified, propose dataset, model, timed calibration, runtime/cost ceiling, checkpointing and stop rule before launch. Do not infer that the existing baseline has reached an architecture ceiling.
