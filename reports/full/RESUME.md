# Paused full-data MiniLM experiment

The local job was suspended with `SIGSTOP`; it is not training while suspended.
The progress-update automation `minilm-training-updates` is also paused.

## Saved to disk

- Original IMDb dataset: `data/raw/imdb/dataset.zip`.
- Complete tokenized, movie-disjoint splits: `data/full/`.
- Pretrained MiniLM: `models/base/`.
- All 64,405 before-training test scores: `reports/full/before.npy`.
- Before-training example scores: `reports/full/before-examples.json`.
- Pilot model and completed pilot report remain unchanged.
- Pause details: `reports/full/pause-state.json`.

The full-data job had entered training but had not reached its first 500-batch
checkpoint. **There is no full-data training checkpoint on disk yet.** Its exact
current model and optimizer state are retained in the suspended process's memory.

## Resume while the process still exists

First verify that PID **76720** is still the suspended Python process running
`scripts/full_training.py`; do not signal an unrelated process if the PID was reused.
Then send `SIGCONT`:

```sh
ps -p 76720 -o pid=,stat=,command=
kill -CONT 76720
```

The process continues from its suspended state. Do not start a second training job.
Re-enable the update automation if progress notifications are desired.

## Resume after a reboot or process loss

Data preparation and baseline evaluation do not need to run again. If no
`models/minilm-full/last.pt` exists, launch:

```sh
.venv/bin/python scripts/start_full_training.py
```

This restarts **training** from pretrained weights and reuses the completed baseline.
The initial unsaved training steps are lost. If `last.pt` does exist, use:

```sh
.venv/bin/python scripts/start_full_training.py --resume
```

The launcher checks for an existing PID and refuses a duplicate launch; if that PID
has been reused, inspect and correct `reports/full/process.json` before retrying.
