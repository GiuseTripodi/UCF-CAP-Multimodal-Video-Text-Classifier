# Frozen in Time - Multimodal Video Anomaly Detection

This project performs **multimodal video anomaly detection** using a pretrained video foundation model (TimeSformer or ViViT) and a pretrained text encoder (DistilBERT). The pipeline fine-tunes a small subset of the video backbone while keeping the text encoder frozen, then fuses video + caption embeddings for classification.

## What It Does

- **Inputs**: video clips (as frame folders) + text captions
- **Backbones**: pretrained TimeSformer or ViViT (HuggingFace), DistilBERT
- **Fusion**: concat/add/cross-attention over video + text embeddings
- **Output**: anomaly class logits (multiclass)

## Data Format

The loader expects a CSV (or pickle) with these columns:

- `videoID`: unique id
- `caption`: text description
- `video_path`: path to a directory of frames (`*.jpg`)
- `label`: class label (string)

Example CSV row:

```
videoID,caption,video_path,label
v_001,person running,/path/to/frames/v_001,normal
v_002,person falling,/path/to/frames/v_002,anomaly
```

The project expects the UCF-CAP split files under:

```
<save_dir>/UcfCap/train_dataset.csv
<save_dir>/UcfCap/val_dataset.csv
<save_dir>/UcfCap/test_dataset.csv
```

`<save_dir>` defaults to `data/` when using the provided scripts.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/multi_vit/train_multi_vit.py \
  --config /Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json \
  --save_dir /Users/user/PycharmProjects/frozen-in-time/data \
  --name my_run
```

Key config knobs (in `configs/ucf-cap.json`):

- `model.video_model_name`: `facebook/timesformer-base-finetuned-k400` or `google/vivit-b-16x2-kinetics400`
- `model.text_model_name`: `distilbert-base-uncased`
- `model.fusion_method`: `concat`, `add`, or `cross_attention`
- `model.freeze_video_backbone`: `true` to only fine-tune the last layers
- `model.trainable_layers`: number of transformer layers to unfreeze
- `data_loader.sampling_method`: `interpolate` or `uniform`
- `trainer.num_frames`: number of frames sampled per clip

## Evaluate

```bash
python src/multi_vit/eval_multimodal.py \
  --config /Users/user/PycharmProjects/frozen-in-time/configs/ucf-cap.json \
  --save_dir /Users/user/PycharmProjects/frozen-in-time/data \
  --checkpoint /Users/user/PycharmProjects/frozen-in-time/data/models/best_model_.pth
```

## Sanity Check (No Data Required)

```bash
python scripts/sanity_check.py --skip-model
```

## How It Works (High Level)

1. **Frame sampling**: each video is represented by `num_frames` frames from its `video_path` folder.
2. **Video backbone**: pretrained TimeSformer/ViViT produces a CLS embedding for the video.
3. **Text backbone**: pretrained DistilBERT produces a CLS embedding for the caption.
4. **Fusion**: video + text embeddings are fused (concat/add/cross-attention).
5. **Classifier**: MLP head predicts anomaly classes.

## Notes

- Pretrained backbones are downloaded from HuggingFace on first run.
- To fine-tune more of the video backbone, increase `model.trainable_layers` or set `model.freeze_video_backbone` to `false`.
- `train_dataset.csv`, `val_dataset.csv`, and `test_dataset.csv` must be present under `<save_dir>/UcfCap/`.

