# DTMEL

Official Repository of "Dual-Tower Multimodal Entity Linking with Deep Interaction".

## Structure

```text
DTMEL/
  main.py
  run.sh
  requirements.txt
  config/
    wikimel_dual_tower.yaml
    wikidiverse_dual_tower.yaml
    richpediamel_dual_tower.yaml
  codes/
    model/
      modeling_dual_tower.py
      lightning_dual_tower.py
    utils/
      dataset.py
      functions.py
      gpu_monitor.py
```

## Setup

Install the Python dependencies in your own environment:

```bash
pip install -r requirements.txt
```

The configs use `openai/clip-vit-base-patch32` by default. If the model is stored locally, replace `pretrained_model` with that local path.

## Data Paths

Data is not bundled in this folder. Before training, edit one config file under `config/` so that these fields point to your local dataset files:

```yaml
data:
  kb_img_folder: 'data/<Dataset>/kb_image'
  mention_img_folder: 'data/<Dataset>/mention_images'
  qid2id: 'data/<Dataset>/qid2id.json'
  entity: 'data/<Dataset>/kb_entity.json'
  train_file: 'data/<Dataset>/<train>.json'
  dev_file: 'data/<Dataset>/<dev>.json'
  test_file: 'data/<Dataset>/<test>.json'
```

## Run

```bash
python main.py --config config/wikidiverse_dual_tower.yaml
```

or:

```bash
GPU=0 CONFIG=./config/wikidiverse_dual_tower.yaml bash run.sh
```

Training logs and checkpoints are written to `runs/` at runtime.

## Model Summary

The model uses CLIP as a multimodal encoder, projects text and image representations into a shared dimension, applies MoE or dense sequence projection with optional cross-modal attention, and trains a dual-tower matcher with InfoNCE-style contrastive loss.
