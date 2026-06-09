# DTMEL Cleanup Manifest

## Included

- `main.py`: training and test entry point.
- `run.sh`: minimal shell runner with configurable GPU and config path.
- `codes/model/modeling_dual_tower.py`: dual-tower encoder, fusion tower, MoE/dense projectors, matcher.
- `codes/model/lightning_dual_tower.py`: PyTorch Lightning training, validation, test, loss, logging, and optimizer logic.
- `codes/utils/dataset.py`: dataset loading, preprocessing cache, image handling, collators, and dataloaders.
- `codes/utils/functions.py`: config loader and CLI parser.
- `codes/utils/gpu_monitor.py`: Lightning callback for peak GPU memory logging.
- `config/*.yaml`: cleaned configuration templates for WikiMEL, WikiDiverse, and RichpediaMEL.
- `requirements.txt`: Python package dependencies.
- `README.md`: setup, data path, run, and model summary documentation.

## Excluded

- `result/`, `runs/`, and `experiments/`: generated results, logs, and experiment batches.
- `paper-towerv5IPM-majorrevise/`: manuscript materials.
- `unused/`: abandoned analysis and old scripts.
- `case_study/` and `tools/`: non-core analysis/figure utilities.
- `.idea/`, `.codex/`, `.git/`, `__pycache__/`, and `*.pyc`: IDE, agent, VCS, and bytecode/cache files.
- `*_originVersion.py`: historical source snapshots.
- Dataset files and pretrained model files.

## Cleanup Performed

- Removed copied historical implementation blocks and pure comment-only lines from the Python core copy.
- Removed temporary hard-coded `/mnt/...` dataset assignments from `dataset.py`.
- Replaced personal absolute model paths in configs with `openai/clip-vit-base-patch32`.
- Replaced external dataset paths in configs with local template paths under `data/<Dataset>/...`.
- Added package `__init__.py` files so imports remain explicit.
