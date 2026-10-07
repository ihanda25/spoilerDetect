# Validation threshold comparison

Thresholds selected on validation, not unbiased test results. No training performed.

| Checkpoint | Operating point | Threshold | Precision | Recall | F1 | False positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| review_epoch1 | default | 0.5000 | 0.659 | 0.487 | 0.560 | 4388 |
| review_epoch1 | max_f1 | 0.3565 | 0.527 | 0.686 | 0.596 | 10711 |
| review_epoch1 | recall_60% | 0.4204 | 0.585 | 0.600 | 0.592 | 7387 |
| review_epoch1 | recall_70% | 0.3457 | 0.517 | 0.700 | 0.595 | 11338 |
| review_epoch1 | recall_80% | 0.2491 | 0.452 | 0.800 | 0.578 | 16805 |
| review_epoch1 | recall_90% | 0.1443 | 0.383 | 0.900 | 0.538 | 25133 |
| review_epoch2 | default | 0.5000 | 0.727 | 0.422 | 0.534 | 2748 |
| review_epoch2 | max_f1 | 0.2158 | 0.532 | 0.697 | 0.603 | 10628 |
| review_epoch2 | recall_60% | 0.2891 | 0.597 | 0.600 | 0.598 | 7038 |
| review_epoch2 | recall_70% | 0.2134 | 0.530 | 0.700 | 0.603 | 10788 |
| review_epoch2 | recall_80% | 0.1495 | 0.462 | 0.800 | 0.586 | 16139 |
| review_epoch2 | recall_90% | 0.0936 | 0.388 | 0.900 | 0.542 | 24632 |
| plot_epoch1 | default | 0.5000 | 0.573 | 0.540 | 0.556 | 6988 |
| plot_epoch1 | max_f1 | 0.3922 | 0.513 | 0.649 | 0.573 | 10719 |
| plot_epoch1 | recall_60% | 0.4404 | 0.540 | 0.600 | 0.568 | 8884 |
| plot_epoch1 | recall_70% | 0.3407 | 0.483 | 0.700 | 0.572 | 13014 |
| plot_epoch1 | recall_80% | 0.2321 | 0.428 | 0.800 | 0.558 | 18564 |
| plot_epoch1 | recall_90% | 0.1285 | 0.374 | 0.900 | 0.528 | 26164 |
