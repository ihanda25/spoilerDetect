"""CUDA-only, validation-first evaluation of the pinned IMDb RoBERTa checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, roc_auc_score)

MODEL_ID = "Zritze/imdb-spoiler-robertaOrigDatasetLR1"
MODEL_REVISION = "56fee120f8495ccfc5001e3fbd1478656d17001a"
MAX_LENGTH = 512
EXPECTED_ROWS = {"validation": 63836, "test": 64405}
ROW_KEYS = {"source_line", "movie_id", "text", "label"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def max_f1_threshold(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Validation labels/scores must be nonempty equal-length vectors")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Validation must contain both binary labels")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    p, r, thresholds = precision_recall_curve(labels, scores)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    # Deliberately mirrors tune_validation_thresholds.py, including np.argmax tie behavior.
    return float(thresholds[int(np.argmax(f))])


def fingerprint(model_revision, input_sha256, batch_size, dtype="float32", runtime=None):
    payload = dict(model_id=MODEL_ID, model_revision=model_revision, input_sha256=input_sha256,
                   max_length=MAX_LENGTH, dtype=dtype, batch_size=int(batch_size),
                   score="softmax_class_1", padding="dynamic",runtime=runtime,
                   evaluator_sha256=sha256_file(Path(__file__)))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def validate_manifest(manifest, data_dir: Path, verify_hashes=True):
    """Validate manifest file hashes/counts and expose split metadata without opening rows."""
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not {"validation", "test"}.issubset(splits):
        raise ValueError("manifest.json must bind validation and test splits")
    resolved = {}
    for split in ("validation", "test"):
        meta = splits[split]
        if not isinstance(meta, dict) or not isinstance(meta.get("file"), str):
            raise ValueError(f"manifest split {split} must specify file, sha256 and rows")
        path = (data_dir / meta["file"]).resolve()
        if data_dir.resolve() not in path.parents:
            raise ValueError("Manifest file path escapes data directory")
        if not path.is_file() or not meta.get("sha256"):
            raise ValueError(f"Missing {split} input or SHA256 in manifest")
        if verify_hashes is True or split in verify_hashes:
            if sha256_file(path) != meta["sha256"]:
                raise ValueError(f"{split} input SHA256 does not match manifest")
        if not isinstance(meta.get("rows"), int) or meta["rows"] < 1:
            raise ValueError(f"Invalid {split} row count in manifest")
        resolved[split] = (path, meta)
    if resolved["validation"][0] == resolved["test"][0]:
        raise ValueError("Validation and test must be distinct split files")
    if "train" in splits:
        if not isinstance(splits["train"], dict):
            raise ValueError("Invalid train split metadata")
        train_file = splits["train"].get("file")
        if train_file and train_file in {splits[s]["file"] for s in ("validation", "test")}:
            raise ValueError("Train and evaluation splits must use distinct files")
    return resolved


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != ROW_KEYS:
                raise ValueError(f"{path.name} row {line_no} must have exactly {sorted(ROW_KEYS)}")
            if (not isinstance(row["source_line"], int) or
                    not isinstance(row["movie_id"], str) or not row["movie_id"] or
                    not isinstance(row["text"], str) or row["label"] not in (0, 1)):
                raise ValueError(f"Invalid typed field in {path.name} row {line_no}")
            yield row


def load_saved_metrics(data_dir: Path, split: str):
    path = data_dir / f"minilm-{split}-results.json"
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())
    if split == "validation" and isinstance(metrics, dict) and "review_epoch2" in metrics:
        return metrics["review_epoch2"]
    return metrics


def run_split(split, path, meta, tokenizer, model, out_dir, batch_size, device, progress_every,runtime=None):
    input_hash = meta["sha256"]
    fp, fp_payload = fingerprint(MODEL_REVISION, input_hash, batch_size,runtime=runtime)
    prefix = out_dir / f"roberta-{split}"
    scores_path, labels_path, lines_path = (prefix.with_suffix(s) for s in (".scores.npy", ".labels.npy", ".source-lines.npy"))
    state_path = prefix.with_suffix(".progress.json")
    n = meta["rows"]
    if n != EXPECTED_ROWS[split]:
        raise ValueError(f"{split} manifest count must be {EXPECTED_ROWS[split]}, got {n}")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fp or state.get("fingerprint_payload") != fp_payload or state.get("count") != n:
            raise ValueError(f"Refusing incompatible {split} resume cache; remove {state_path} to restart")
        done = int(state["completed"])
        truncated = int(state.get("truncation_count", 0))
        if not (scores_path.exists() and labels_path.exists() and lines_path.exists()):
            raise ValueError(f"Incomplete cache files for {split}")
    else:
        done = 0
        truncated = 0
        for file in (scores_path, labels_path, lines_path):
            if file.exists():
                file.unlink()
        np.lib.format.open_memmap(scores_path, mode="w+", dtype="float32", shape=(n,)).flush()
        np.lib.format.open_memmap(labels_path, mode="w+", dtype="int8", shape=(n,)).flush()
        np.lib.format.open_memmap(lines_path, mode="w+", dtype="int64", shape=(n,)).flush()
        atomic_json(state_path, dict(fingerprint=fp, fingerprint_payload=fp_payload, count=n, completed=0))
    # Avoid trusting cache progress beyond the number of records available.
    if done < 0 or done > n:
        raise ValueError(f"Invalid resume offset for {split}")
    scores = np.load(scores_path, mmap_mode="r+")
    labels = np.load(labels_path, mmap_mode="r+")
    lines = np.load(lines_path, mmap_mode="r+")
    if any(arr.shape != (n,) for arr in (scores,labels,lines)):
        raise ValueError(f'Wrong cached array shape for {split}')
    started = time.monotonic()
    seen = 0
    text_batch, label_batch, line_batch = [], [], []
    import torch
    with torch.inference_mode():
        for row in iter_rows(path):
            idx = seen
            seen += 1
            if idx < done:
                if labels[idx] != row['label'] or lines[idx] != row['source_line']:
                    raise ValueError(f'Cached label/source order mismatch in {split}')
                continue
            text_batch.append(row["text"])
            label_batch.append(row["label"])
            line_batch.append(row["source_line"])
            if len(text_batch) == batch_size or seen == n:
                raw = tokenizer(text_batch, truncation=False, padding=False, add_special_tokens=True)
                truncated += sum(len(ids) > MAX_LENGTH for ids in raw["input_ids"])
                encoded = tokenizer(text_batch, truncation=True, max_length=MAX_LENGTH,
                                    padding=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                probs = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
                if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
                    raise ValueError('Invalid model probability')
                start, end = seen - len(text_batch), seen
                scores[start:end] = probs.astype(np.float32)
                labels[start:end] = label_batch
                lines[start:end] = line_batch
                scores.flush(); labels.flush(); lines.flush()
                atomic_json(state_path, dict(fingerprint=fp, fingerprint_payload=fp_payload,
                                             count=n, completed=end, truncation_count=truncated))
                text_batch.clear(); label_batch.clear(); line_batch.clear()
                if end % progress_every < batch_size or end == n:
                    elapsed = time.monotonic() - started
                    rate = (end - done) / max(elapsed, 1e-9)
                    eta = (n - end) / max(rate, 1e-9)
                    print(f"{split}: {end:,}/{n:,} records, {rate:.1f}/s, elapsed {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)
    if seen != n:
        raise ValueError(f"Manifest count {n} differs from {split} JSONL count {seen}")
    if int(np.sum(labels)) != meta.get('spoilers',int(np.sum(labels))):
        raise ValueError(f'Label class counts disagree with manifest for {split}')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Invalid cached results for {split}")
    np.savez_compressed(out_dir / f"roberta-{split}-scores.npz", scores=np.asarray(scores),
                        labels=np.asarray(labels), source_lines=np.asarray(lines))
    return np.asarray(scores), np.asarray(labels), np.asarray(lines), truncated, fp, time.monotonic() - started


def operating_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    return dict(threshold=float(threshold), accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                f1=float(f1_score(labels, predicted, zero_division=0)),
                false_positives=int(cm[0, 1]), false_negatives=int(cm[1, 0]),
                confusion_matrix=cm.tolist(), roc_auc=float(roc_auc_score(labels, scores)),
                average_precision=float(average_precision_score(labels, scores)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU inference")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    split_inputs = validate_manifest(manifest, args.data_dir, verify_hashes={"validation"})
    device = torch.device("cuda")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True,trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                                num_labels=2, use_safetensors=True,trust_remote_code=False)
    if model.config.num_labels != 2 or model.config.id2label.get(1,'').upper() != 'SPOILER':
        raise ValueError('Unverified RoBERTa spoiler class')
    model.float().to(device).eval()
    common = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    import transformers
    common["transformers"] = transformers.__version__
    common["model_id"] = MODEL_ID
    common["model_revision"] = MODEL_REVISION
    common["precision"] = "float32"
    common["batch_size"] = args.batch_size
    runtime={k:common[k] for k in ('gpu','torch','transformers')}
    # Validation must be fully scored and selection persisted before test JSONL is opened.
    vpath, vmeta = split_inputs["validation"]
    vs, vy, vl, vtrunc, vfp, vtime = run_split("validation", vpath, vmeta, tokenizer, model,
                                               args.out, args.batch_size, device, 512,runtime)
    threshold = max_f1_threshold(vy, vs)
    selection = dict(threshold=threshold, method="validation precision_recall_curve maximum F1; np.argmax",
                     validation_sha256=vmeta["sha256"], validation_count=len(vy),
                     validation_cache_fingerprint=vfp, selected_before_test_load=True)
    selection_path=args.out / 'threshold-selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Existing frozen threshold differs; refusing to change selection')
    atomic_json(selection_path, selection)
    print('VALIDATION COMPLETE: frozen threshold',threshold,flush=True)
    validation_report = dict(split="validation", rows=len(vy), truncation_count=int(vtrunc),
                             max_f1_threshold=threshold,
                             input_sha256=vmeta["sha256"], cache_fingerprint=vfp,
                             label_counts={"0": int(np.sum(vy == 0)), "1": int(np.sum(vy == 1))},
                             default_metrics=operating_metrics(vy, vs, 0.5),
                             tuned_metrics=operating_metrics(vy, vs, threshold),
                             minilm=load_saved_metrics(args.data_dir, "validation"), elapsed_seconds=vtime)
    atomic_json(args.out / "validation-results.json", {**common, **validation_report})
    # Only now resolve/open test rows.
    tpath, tmeta = split_inputs["test"]
    if sha256_file(tpath) != tmeta["sha256"]:
        raise ValueError("test input SHA256 does not match manifest")
    ts, ty, tl, ttrunc, tfp, ttime = run_split("test", tpath, tmeta, tokenizer, model,
                                               args.out, args.batch_size, device, 512,runtime)
    report = dict(**common, split="test", rows=len(ty), truncation_count=int(ttrunc),
                  label_counts={"0": int(np.sum(ty == 0)), "1": int(np.sum(ty == 1))},
                  default_threshold_metrics=operating_metrics(ty, ts, 0.5),
                  frozen_validation_threshold=threshold,
                  tuned_threshold_metrics=operating_metrics(ty, ts, threshold),
                  validation_selection=selection, test_sha256=tmeta["sha256"],
                  test_cache_fingerprint=tfp, minilm=load_saved_metrics(args.data_dir, "test"),
                  elapsed_seconds=time.monotonic() - started,
                  split_processing_seconds={"validation": vtime, "test": ttime},
                  warning="Author-level train split provenance is unknown; potential contamination cannot be ruled out.")
    atomic_json(args.out / "test-results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("minilm",)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
