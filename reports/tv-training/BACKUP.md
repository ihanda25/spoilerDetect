# Checkpoint backup status

Training completed October 6, 2026. Comparison reports and prediction caches are saved locally. Final model ZIPs remain in the connected Colab runtime under `/content/tv-sentence-epoch1.zip` and `/content/tv-context-epoch1.zip`. Their sizes and SHA256 values are recorded in model-archives.json.

Browser downloads of the large ZIPs did not produce verified local files. Both ZIPs were split into 64 MiB parts (`.part00` through `.part06`), with per-part hashes in `/content/tv-model-parts.json`. Concatenate each model's parts in numeric order and verify the original full archive SHA256 before extracting. Never restart training solely to retry exports.

The training and second epoch are stopped. Keep this runtime's files until model backups are verified; deleting the runtime destroys its temporary files. The existing Drive notebook displays an automatic-saving conflict, so the reproducible experiment notebook and launch script are saved in this project. No claim is made that the appended cells are saved on Drive.
