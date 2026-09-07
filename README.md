# UCF-CAP Multimodal Video-Text Classifier

Multimodal video classification for the UCF-CAP dataset. The model combines:

- a pretrained video backbone (`TimeSformer` or `ViViT`)
- a pretrained text encoder (`DistilBERT`)
- a fusion head for final class prediction

## Project layout

- `src/multi_vit/train_multi_vit.py` — training entry point
- `src/multi_vit/eval_multimodal.py` — evaluation entry point
- `model/multi_vit.py` — multimodal model
- `data_loader/ucf_cap_loader.py` — dataset and batching logic
- `src/trainers/trainer_multimodal.py` — training loop and checkpointing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Place the dataset splits under `data/UcfCap/`:

- `train_dataset.csv`
- `val_dataset.csv`
- `test_dataset.csv`

Each row should include:

- `videoID`
- `caption`
- `video_path` pointing to a folder of `*.jpg` frames
- `label`

## Train

```bash
python src/multi_vit/train_multi_vit.py \
  --config configs/ucf-cap.json \
  --save_dir data \
  --name my_run
```

## Evaluate

```bash
python src/multi_vit/eval_multimodal.py \
  --config configs/ucf-cap.json \
  --save_dir data \
  --checkpoint data/models/best_model.pth
```

## Quick sanity check

```bash
python scripts/sanity_check.py --skip-model
```

## Outputs

- checkpoints: `data/models/`
- logs: `data/logs/`
- training curves/history: saved next to the checkpoints

## Notes

- The default config is `configs/ucf-cap.json`.
- Edit the config if you want to change model backbones, frame count, or fusion type.
- Hugging Face weights are downloaded on first run.

