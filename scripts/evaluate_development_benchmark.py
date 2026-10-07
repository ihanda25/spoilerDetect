"""Offline CPU inference scored ONLY against proposals; human labels are ignored."""
import hashlib
import json
import math
import time
from pathlib import Path
from build_development_benchmark import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/development-benchmark'
THRESHOLD = .21579217910766602


def summarize(labels, scores):
    assert len(labels) == len(scores)
    assert all(y in (0, 1) for y in labels)
    assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
    tp = sum(y == 1 and s >= THRESHOLD for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= THRESHOLD for y, s in zip(labels, scores))
    fn = sum(y == 1 and s < THRESHOLD for y, s in zip(labels, scores))
    tn = len(labels)-tp-fp-fn
    return dict(n=len(labels), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def invariants():
    assert summarize([0, 1], [0., 1.])['f1'] == 1.
    assert summarize([1, 0], [0., 1.])['f1'] == 0.
    assert summarize([1], [THRESHOLD])['tp'] == 1
    assert summarize([1], [math.nextafter(THRESHOLD, 0)])['fn'] == 1
    assert summarize([], [])['n'] == 0
    assert summarize([0], [0.])['precision'] == 0.
    for bad in (float('nan'), float('inf'), -0.1, 1.1):
        try:
            summarize([0], [bad])
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid score accepted')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    invariants()
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    source = ROOT / 'data/development-benchmark/examples.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    validate(rows)
    cp = ROOT / 'models/minilm-full/checkpoint.pt'
    checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
    assert checkpoint['completed_epochs'] == 2
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/base', local_files_only=True, num_labels=2)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    assert tokenizer.truncation_side == 'right'
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset+4]
            inputs = tokenizer([r['text'] for r in batch], truncation=True, max_length=512, padding=True, return_tensors='pt')
            logits = model(**inputs).logits
            scores = logits.softmax(-1)[:, 1].tolist()
            for i, (row, score) in enumerate(zip(batch, scores)):
                full = tokenizer(row['text'], truncation=False, verbose=False)['input_ids']
                visible = inputs['input_ids'][i][inputs['attention_mask'][i].bool()].tolist()
                expected = full if len(full) <= 512 else full[:511] + [tokenizer.sep_token_id]
                assert visible == expected
                outputs.append(dict(id=row['id'], surface=row['surface'], challenge=row['challenge'],
                    proposed_label=row['proposed_label'], score=score, prediction=int(score >= THRESHOLD),
                    character_count=len(row['text']), whitespace_words=len(row['text'].split()),
                    full_token_count=len(full), visible_token_count=len(visible),
                    removed_token_count=len(full)-len(visible), truncated=len(full)>512,
                    visible_token_ids=visible, visible_decoded_text=tokenizer.decode(visible),
                    timestamp=row.get('timestamp'), diagnostic_pair_id=row.get('diagnostic_pair_id')))
            print(f'CPU inference {offset+len(batch)}/{len(rows)}', flush=True)
    assert len(outputs) == len(rows)
    assert all(math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in outputs)
    # Long negative controls must have precisely the same retained tokens as late-reveal reviews.
    pairs = []
    for work_id in sorted({r['work_id'] for r in rows}):
        by_tag = {o['challenge']: o for r, o in zip(rows, outputs) if r['work_id'] == work_id and r['surface'] == 'review'}
        late, early, control = (by_tag[k] for k in ('late_reveal', 'early_reveal', 'long_craft_only'))
        assert late['visible_token_ids'] == control['visible_token_ids']
        assert abs(late['score']-control['score']) < 1e-5
        assert early['truncated'] and late['truncated'] and control['truncated']
        pairs.append(dict(work_id=work_id, early_score=early['score'], late_score=late['score'],
                          control_score=control['score'], late_control_identical_visible_input=True))
    groups = {}
    for surface in ['all', 'headline', 'comment', 'review', 'youtube_transcript']:
        selected = [r for r in outputs if r['proposed_label'] is not None and (surface == 'all' or r['surface'] == surface)]
        groups[surface] = summarize([r['proposed_label'] for r in selected], [r['score'] for r in selected])
    result = dict(purpose='Synthetic development challenge agreement, NOT production accuracy or human-reviewed validation',
                  threshold=THRESHOLD, threshold_fitted=False, input_policy='Text only; context is annotation metadata, not model input',
                  max_length=512, truncation_side='right', device='cpu', batch_size=4, threads=2,
                  label_mapping={'0': 'safe', '1': 'spoiler'}, tokenizer='models/base',
                  base_file_sha256={name: sha(ROOT / 'models/base' / name) for name in
                                    ('config.json', 'vocab.txt', 'tokenizer_config.json', 'special_tokens_map.json')},
                  checkpoint=str(cp.relative_to(ROOT)), completed_epochs=2, checkpoint_sha256=sha(cp),
                  examples_sha256=sha(source), torch_version=torch.__version__, transformers_version=transformers.__version__,
                  label_source='proposed_label only; annotation.human_label is intentionally ignored',
                  examples=len(rows), unambiguous=sum(r['proposed_label'] is not None for r in rows),
                  ambiguous_excluded=sum(r['proposed_label'] is None for r in rows),
                  human_reviewed=sum(r['annotation']['status'] in ('human_reviewed', 'adjudicated') for r in rows),
                  truncated=sum(r['truncated'] for r in outputs), invariant_checks='passed',
                  synthetic_label_agreement=groups, truncation_diagnostics=pairs, elapsed_seconds=time.monotonic()-start)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    (OUT / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Development challenge run', '',
        f"All {len(rows)} examples are assistant-authored synthetic challenges; {result['human_reviewed']} have recorded human review. These numbers measure agreement with provisional assistant labels only, not representative production accuracy. Human labels are intentionally ignored by this scorer. No test rows, training, paid calls, downloads, or threshold tuning.", '',
        f'CPU epoch-2 inference completed in {result["elapsed_seconds"]:.1f}s. Frozen validation threshold: {THRESHOLD}. Text-only input, right truncation at 512 tokens including special tokens; batch size 4, two CPU threads.', '',
        f"{result['unambiguous']} binary proposals scored; {result['ambiguous_excluded']} ambiguous proposals excluded from all binary metrics. All {len(outputs)} predictions retained. Scores are uncalibrated softmax scores.", '',
        '| Surface | N | TP | FP | FN | TN | Precision | Recall | F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in groups.items():
        lines.append(f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ['', '## Full-review truncation diagnostic', '',
              f"{sum(r['surface'] == 'review' and r['truncated'] for r in outputs)} reviews exceed 512 tokens. Each work has an early reveal, the same reveal at the end, and a long craft-only control. Late-reveal and control retained token IDs are identical and their scores agree within 1e-5. The model therefore cannot observe the decisive late reveal. Repetitive padding is deliberately artificial.", '',
              '| Work | Early reveal score | Late reveal score | Negative control score |', '| --- | ---: | ---: | ---: |']
    for p in pairs:
        lines.append(f"| {p['work_id']} | {p['early_score']:.6f} | {p['late_score']:.6f} | {p['control_score']:.6f} |")
    lines += ['', '## Interpretation and artifacts', '',
        'Four fictional works and templated contrasts are correlated, hand-constructed probes. Three extra copies of the missing-antecedent transcript probe are intentional; all are ambiguous and excluded from metrics. Do not interpret sample prevalence, per-surface differences, or these pooled metrics as traffic estimates. No confidence intervals or generalization claims are justified here.', '',
        'Transcript timestamps are invented passage ranges, not observed video or annotated event spans. No video-level false warnings/hour, temporal localization, real ASR robustness, or viewer-progress accuracy is measured. Fictional story context helps annotators; this review-only model does not receive it.', '',
        '`predictions.jsonl` retains lengths, truncation, visible token IDs/decoded text and every score. `results.json` records checkpoint/data hashes, environment and counts. `ANNOTATION.md` defines the pending human-review workflow. `annotation.schema.json` describes each row.', '',
        'Checks passed: finite bounded scores, threshold equality and just-below boundary, perfect/inverted/empty/no-positive cases, invalid-score rejection, row counts/provenance, timestamp ordering, exact right-truncated token input, and late/control input-score equivalence.', '',
        'Reproduce from repository root (offline):', '',
        '```sh', '.venv/bin/python scripts/build_development_benchmark.py',
        'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/evaluate_development_benchmark.py', '```', '',
        'Use human review to revise the policy and collect independent real development data before drawing product conclusions. Keep a future test set sealed and do not tune the frozen threshold on this pack.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
