"""No-training, resumable HF product evaluation; run --help for CLI.

One JSONL must contain both dev and test to enforce group disjointness. Required
fields: id/text/surface/group_id/split (strings), label (0/1/null),
label_provenance (human/corpus/synthetic), review_status
(pending_human_review/human_reviewed/adjudicated/corpus_labeled).
Groups must encompass work/franchise AND source documents; merge connected groups
upstream if a source covers multiple works. IDs/groups are globally unique keys,
not inferred from titles. Null means uncertain/unlabeled and is never binary gold.
Human provenance is an attestation, not something software can independently prove.

Human-reviewed rows also require reviewer_id and timezone-aware ISO reviewed_at.
review_rationale and other additional fields are preserved in the source JSONL.
Human policy requires completed review of every row in the target split and uses
only human + human_reviewed/adjudicated binary labels. Corpus
policy uses only corpus + corpus_labeled binary labels and is explicitly NOT
product gold. Synthetic labels are never scoring targets. Predictions ignore labels.
All commands use the exact same complete dataset bytes. Any edit invalidates caches
and selection. Repeated test evaluation reuses saved predictions; no threshold fit.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = 1
SURFACES = {'headline', 'comment', 'review', 'youtube_transcript'}
# Root files only. Never download pickle weights, training_args or nested checkpoints.
HUB_ALLOW_PATTERNS = ['config.json', 'model.safetensors', 'model.safetensors.index.json',
    'model-?????-of-?????.safetensors', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.txt', 'vocab.json',
    'merges.txt', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model']
HUB_IGNORE_PATTERNS = ['*/*', '*.bin', '*.pt', '*.pth', 'training_args*', '*checkpoint*']



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_data(path):
    rows = []
    ids, groups, texts = set(), {}, {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {'id', 'text', 'surface', 'group_id', 'split', 'label', 'label_provenance', 'review_status'}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f'Row {line_no}: missing required fields')
        for key in ('id', 'text', 'surface', 'group_id', 'split'):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'Row {line_no}: invalid {key}')
        if row['id'] in ids:
            raise ValueError('Duplicate id: ' + row['id'])
        ids.add(row['id'])
        if row['split'] not in ('dev', 'test') or row['surface'] not in SURFACES:
            raise ValueError('Unsupported split or surface')
        if row['label'] is not None and (type(row['label']) is not int or row['label'] not in (0, 1)):
            raise ValueError('label must be integer 0, 1 or null')
        if row['label_provenance'] not in ('human', 'corpus', 'synthetic'):
            raise ValueError('Unsupported label_provenance')
        if row['review_status'] not in ('pending_human_review', 'human_reviewed', 'adjudicated', 'corpus_labeled'):
            raise ValueError('Unsupported review_status')
        reviewed = row['review_status'] in ('human_reviewed', 'adjudicated')
        synthetic = row['label_provenance'] == 'synthetic' or row.get('is_synthetic', False) or row.get('source_type') in ('synthetic', 'assistant_authored_synthetic')
        if reviewed and (row['label_provenance'] != 'human' or synthetic):
            raise ValueError('Synthetic/nonhuman provenance cannot be passed off as human-reviewed gold')
        if reviewed:
            if not isinstance(row.get('reviewer_id'), str) or not row['reviewer_id'].strip():
                raise ValueError('Human review requires nonempty reviewer_id')
            try:
                reviewed_at = datetime.fromisoformat(row.get('reviewed_at', '').replace('Z', '+00:00'))
                if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
                    raise ValueError('Missing time zone')
            except (ValueError, TypeError, AttributeError):
                raise ValueError('Human review requires timezone-aware ISO reviewed_at') from None
        if row['review_status'] == 'corpus_labeled' and row['label_provenance'] != 'corpus':
            raise ValueError('corpus_labeled requires corpus provenance')
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('Group leakage across dev/test: ' + row['group_id'])
        normalized = ' '.join(row['text'].casefold().split())
        if texts.setdefault(normalized, row['split']) != row['split']:
            raise ValueError('Exact normalized text leakage across dev/test')
        rows.append(row)
    if {r['split'] for r in rows} != {'dev', 'test'}:
        raise ValueError('Provide the complete dataset containing both dev and test')
    return rows, file_hash(path)


def eligible(row, policy):
    if row['label'] is None:
        return False
    if policy == 'human':
        return row['label_provenance'] == 'human' and row['review_status'] in ('human_reviewed', 'adjudicated') and not row.get('is_synthetic', False) and row.get('source_type') not in ('synthetic', 'assistant_authored_synthetic')
    if policy == 'corpus':
        return row['label_provenance'] == 'corpus' and row['review_status'] == 'corpus_labeled'
    raise ValueError('Unknown label policy')


def require_split_reviews(rows, split, policy):
    """Uncertain is reviewed but not binary gold; pending test rows close the gate."""
    for row in rows:
        if row['split'] != split:
            continue
        if policy == 'human' and ('group_review_status' in row or 'eligible_for_evaluation' in row):
            if row.get('group_review_status') != 'verified' or row.get('eligible_for_evaluation') is not True:
                raise ValueError(f'{split} is closed: source/work group verification is pending')
        probe = dict(row, label=0)
        if not eligible(probe, policy):
            raise ValueError(f'{split} is closed: every {split} row must complete review under the selected label policy')


def window_ranges(n_tokens, capacity, overlap, mode):
    """Half-open content-token spans; overlap is overlap, not advance distance."""
    if n_tokens < 0 or capacity < 1 or not 0 <= overlap < capacity or mode not in ('truncate', 'window'):
        raise ValueError('Invalid window configuration')
    if mode == 'truncate' or n_tokens <= capacity:
        return [(0, min(n_tokens, capacity))]
    spans = []
    start = 0
    while True:
        end = min(start + capacity, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            return spans
        start = end - overlap


def aggregate(scores):
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Expected nonempty finite window scores in [0,1]')
    return max(scores)


def metrics(labels, scores, threshold):
    if len(labels) != len(scores) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('Invalid labels/scores')
    if not math.isfinite(threshold):
        raise ValueError('Invalid threshold')
    if scores:
        aggregate(scores)
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    fn = sum(y == 1 and s < threshold for y, s in zip(labels, scores))
    tn = len(labels) - tp - fp - fn
    return dict(n=len(labels), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def select_threshold(labels, scores):
    """O(n log n); exact tied-score groups, >= rule. F1 ties pick higher cutoff.

    Candidates include all-positive (0), each observed score, and all-negative
    (nextafter(1,+inf)), so even scores equal to 0 or 1 are handled correctly.
    """
    metrics(labels, scores, 0.)
    if not labels or set(labels) != {0, 1}:
        raise ValueError('Threshold selection requires both binary classes')
    total_pos = sum(labels)
    tp = fp = 0
    best = (0., math.nextafter(1., math.inf))
    pairs = sorted(zip(scores, labels), reverse=True)
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            tp += pairs[i][1]
            fp += 1-pairs[i][1]
            i += 1
        f1 = 2*tp/(total_pos+tp+fp)
        best = max(best, (f1, score))
    best = max(best, (2*total_pos/(total_pos+len(labels)), 0.))
    return best[1], metrics(labels, scores, best[1])


def seal(obj):
    obj = dict(obj)
    obj['integrity_sha256'] = digest(obj)
    return obj


def read_sealed(path):
    obj = json.loads(Path(path).read_text())
    claimed = obj.pop('integrity_sha256', None)
    if claimed != digest(obj):
        raise ValueError('Artifact integrity mismatch: ' + str(path))
    return obj


def cache_header(dataset_hash, config, split, selection_hash=None):
    return dict(version=VERSION, dataset_sha256=dataset_hash, config=config,
                config_sha256=digest(config), split=split, selection_sha256=selection_hash)


def read_cache(path, expected=None):
    obj = read_sealed(path)
    header = obj['header']
    if header['version'] != VERSION or header['config_sha256'] != digest(header['config']):
        raise ValueError('Invalid cache config fingerprint/version')
    if expected is not None and header != expected:
        raise ValueError('Cache mismatch: dataset/model/config/split/selection changed; use a new cache path')
    return obj


def validate_predictions(cache, rows, complete=True):
    target = {r['id']: r for r in rows if r['split'] == cache['header']['split']}
    predictions = cache['predictions']
    if not set(predictions) <= target.keys() or (complete and set(predictions) != target.keys()):
        raise ValueError('Cache has missing or foreign records')
    for rid, pred in predictions.items():
        if pred['text_sha256'] != digest(target[rid]['text']):
            raise ValueError('Prediction text mismatch')
        if pred['score'] != aggregate([w['score'] for w in pred['windows']]):
            raise ValueError('Window aggregation mismatch')
    return target


def targets(rows, predictions, split, policy):
    chosen = [r for r in rows if r['split'] == split and eligible(r, policy)]
    return chosen, [r['label'] for r in chosen], [predictions[r['id']]['score'] for r in chosen]


def validate_selection(selection, dataset_hash, config_hash):
    if selection['version'] != VERSION or selection['dataset_sha256'] != dataset_hash or selection['config_sha256'] != config_hash:
        raise ValueError('Selection dataset/config fingerprint mismatch')
    if selection['source_split'] != 'dev' or selection['label_policy'] not in ('human', 'corpus'):
        raise ValueError('Selection must come from reviewed dev or explicitly selected corpus dev')
    if digest(selection['config']) != selection['config_sha256']:
        raise ValueError('Selection config corruption')
    if not 0 <= selection['threshold'] <= math.nextafter(1., math.inf):
        raise ValueError('Invalid frozen threshold')


def resolve_model(args):
    if Path(args.model).is_dir():
        folder = Path(args.model).resolve()
    else:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(args.model, revision=args.revision,
                                       local_files_only=not args.allow_download,
                                       allow_patterns=HUB_ALLOW_PATTERNS,
                                       ignore_patterns=HUB_IGNORE_PATTERNS)).resolve()
    import fnmatch
    patterns = HUB_ALLOW_PATTERNS + (['pytorch_model.bin', 'pytorch_model.bin.index.json',
        'pytorch_model-?????-of-?????.bin'] if Path(args.model).is_dir() else [])
    files = sorted(p for p in folder.iterdir() if p.is_file() and any(fnmatch.fnmatch(p.name, pattern) for pattern in patterns))
    if not files:
        raise ValueError('No local HF model/tokenizer files')
    hashes = {str(p.relative_to(folder)): file_hash(p) for p in files}
    return folder, hashes


def predict(args):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows, dataset_hash = load_data(args.data)
    if args.require_cuda and args.device != 'cuda':
        raise ValueError('--require-cuda requires --device cuda')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    if args.batch_size < 1 or args.threads < 1 or not 3 <= args.max_length <= 512:
        raise ValueError('Invalid batch size/threads/max length (maximum 512)')
    if args.split == 'test' and not args.selection:
        raise ValueError('Test prediction requires a frozen --selection manifest')
    folder, hashes = resolve_model(args)
    config = dict(pipeline_version=VERSION, model_files=hashes,
                  checkpoint_sha256=file_hash(args.checkpoint) if args.checkpoint else None,
                  mode=args.mode, max_length=args.max_length, overlap_tokens=args.stride,
                  aggregation='max', input='text_only', positive_class=args.positive_class,
                  device=args.device, batch_size=args.batch_size, threads=args.threads,
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  dtype='float32', trust_remote_code=False)
    selection = read_sealed(args.selection) if args.selection else None
    if selection:
        validate_selection(selection, dataset_hash, digest(config))
        if args.split == 'test':
            require_split_reviews(rows, 'test', selection['label_policy'])
    header = cache_header(dataset_hash, config, args.split, digest(selection) if selection else None)
    cache = read_cache(args.cache, header) if Path(args.cache).exists() else dict(header=header, predictions={})
    target = validate_predictions(cache, rows, complete=False)
    if len(cache['predictions']) == len(target):
        print(f'Exact cache hit: {len(target)} records; no inference')
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required to preserve character offsets')
    capacity = args.max_length - tokenizer.num_special_tokens_to_add(pair=False)
    window_ranges(0, capacity, args.stride, args.mode)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True, trust_remote_code=False, **({'num_labels': 2} if args.checkpoint else {}))
    if args.checkpoint:
        # Only load a user-supplied trusted local state checkpoint. No optimizer use.
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
    if model.config.num_labels != 2 or args.positive_class not in (0, 1):
        raise ValueError('Requires a two-logit classifier and explicit positive-class index')
    if args.max_length > getattr(model.config, 'max_position_embeddings', args.max_length):
        raise ValueError('max-length exceeds model position capacity')
    model.float().to(args.device).eval()
    for rid, row in target.items():
        if rid in cache['predictions']:
            continue
        encoded = tokenizer(row['text'], add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
        tokens, offsets = encoded['input_ids'], encoded['offset_mapping']
        spans = window_ranges(len(tokens), capacity, args.stride, args.mode)
        windows = []
        with torch.inference_mode():
            for begin in range(0, len(spans), args.batch_size):
                batch_spans = spans[begin:begin+args.batch_size]
                inputs = [tokenizer.prepare_for_model(tokens[a:b], add_special_tokens=True, truncation=False, return_attention_mask=True) for a, b in batch_spans]
                tensors = tokenizer.pad(inputs, padding=True, return_tensors='pt').to(args.device)
                scores = model(**tensors).logits.softmax(-1)[:, args.positive_class].cpu().tolist()
                for (a, b), inp, score in zip(batch_spans, inputs, scores):
                    windows.append(dict(token_start=a, token_end=b,
                        char_start=offsets[a][0] if a < b else 0, char_end=offsets[b-1][1] if a < b else 0,
                        input_token_count=len(inp['input_ids']), score=score))
        cache['predictions'][rid] = dict(text_sha256=digest(row['text']), surface=row['surface'],
            full_content_tokens=len(tokens), covered_content_tokens=spans[-1][1],
            dropped_content_tokens=len(tokens)-spans[-1][1], windows=windows,
            score=aggregate([w['score'] for w in windows]))
        atomic_write(args.cache, seal(cache))
        print(f'{args.split}: {len(cache["predictions"])}/{len(target)} records; {len(windows)} windows', flush=True)


def select(args):
    rows, data_hash = load_data(args.data)
    require_split_reviews(rows, 'dev', args.label_policy)
    candidates = []
    for path in args.cache:
        cache = read_cache(path)
        if cache['header']['split'] != 'dev' or cache['header']['dataset_sha256'] != data_hash:
            raise ValueError('Configuration/threshold selection requires matching dev predictions only')
        validate_predictions(cache, rows)
        chosen, labels, scores = targets(rows, cache['predictions'], 'dev', args.label_policy)
        threshold, result = select_threshold(labels, scores)
        candidates.append(dict(cache=cache, threshold=threshold, metrics=result,
                               ids=[r['id'] for r in chosen]))
    # Deterministic config tie break, independent of caller ordering or test outcomes.
    winner = sorted(candidates, key=lambda c: (-c['metrics']['f1'], c['cache']['header']['config_sha256']))[0]
    cache = winner['cache']
    selection = dict(version=VERSION, source_split='dev', dataset_sha256=data_hash,
        config_sha256=cache['header']['config_sha256'], config=cache['header']['config'],
        dev_cache_sha256=digest(cache), threshold=winner['threshold'], comparator='>=',
        objective='maximum dev F1; threshold ties highest cutoff; config ties lexicographic config hash',
        label_policy=args.label_policy,
        evidence='human-reviewed product dev' if args.label_policy == 'human' else 'corpus-label agreement; NOT product gold',
        selected_ids=winner['ids'], excluded_count=sum(r['split'] == 'dev' for r in rows)-len(winner['ids']),
        metrics=winner['metrics'], candidates=[dict(config_sha256=c['cache']['header']['config_sha256'],
            dev_cache_sha256=digest(c['cache']), threshold=c['threshold'], metrics=c['metrics'])
            for c in sorted(candidates, key=lambda c: c['cache']['header']['config_sha256'])])
    if Path(args.selection).exists():
        if read_sealed(args.selection) != selection:
            raise ValueError('Refusing to overwrite a different frozen selection; use a new experiment path')
    else:
        atomic_write(args.selection, seal(selection))
    print(json.dumps(selection, indent=2))


def evaluate(args):
    rows, data_hash = load_data(args.data)
    cache, selection = read_cache(args.cache), read_sealed(args.selection)
    validate_selection(selection, data_hash, cache['header']['config_sha256'])
    require_split_reviews(rows, 'test', selection['label_policy'])
    if cache['header']['split'] != 'test' or cache['header']['dataset_sha256'] != data_hash or cache['header']['selection_sha256'] != digest(selection):
        raise ValueError('Test predictions must match dataset and frozen selection')
    validate_predictions(cache, rows)
    chosen, labels, scores = targets(rows, cache['predictions'], 'test', selection['label_policy'])
    if not chosen:
        raise ValueError('No eligible test labels under frozen label policy')
    result = dict(version=VERSION, split='test', threshold=selection['threshold'],
        threshold_fitted_on='dev only', selection_sha256=digest(selection), dataset_sha256=data_hash,
        config_sha256=cache['header']['config_sha256'], label_policy=selection['label_policy'],
        evidence='human-reviewed product test' if selection['label_policy'] == 'human' else 'corpus-label agreement; NOT product gold',
        metrics=metrics(labels, scores, selection['threshold']),
        excluded_count=sum(r['split'] == 'test' for r in rows)-len(chosen), per_surface={})
    for surface in sorted(SURFACES):
        selected = [r for r in chosen if r['surface'] == surface]
        result['per_surface'][surface] = metrics([r['label'] for r in selected], [cache['predictions'][r['id']]['score'] for r in selected], selection['threshold'])
    if Path(args.output).exists() and read_sealed(args.output) != result:
        raise ValueError('Refusing to overwrite different locked test results')
    atomic_write(args.output, seal(result))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict', help='Offline by default; cache one completed record atomically at a time')
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True, help='Local HF folder or cached Hub model ID')
    p.add_argument('--revision', default='main')
    p.add_argument('--allow-download', action='store_true', help='Explicitly allow Hub downloads; may be large')
    p.add_argument('--checkpoint', help='Trusted local torch checkpoint with model state dict (existing review epoch2)')
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--mode', choices=['truncate', 'window'], default='truncate')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--stride', type=int, default=128, help='Number of overlapping content tokens, not step size')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--positive-class', type=int, choices=[0, 1], default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cache', required=True)
    p.add_argument('--selection', help='Required before test inference; must match exact dataset and config')
    p.set_defaults(func=predict)
    p = sub.add_parser('select', help='Freeze max-F1 threshold from eligible dev labels only')
    p.add_argument('--data', required=True)
    p.add_argument('--cache', required=True, action='append', help='Repeat to compare configurations on reviewed dev only')
    p.add_argument('--selection', required=True)
    p.add_argument('--label-policy', choices=['human', 'corpus'], default='human')
    p.set_defaults(func=select)
    p = sub.add_parser('evaluate', help='Evaluate test with frozen dev selection; never fit threshold')
    p.add_argument('--data', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
