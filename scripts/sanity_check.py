import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.multi_vit import MultimodalSpaceTimeTransformer


def main(args) -> None:
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.skip_model:
        print("Sanity check: skipped model instantiation.")
        return

    model = MultimodalSpaceTimeTransformer(
        num_classes=args.num_classes,
        video_model_name=args.video_model_name,
        text_model=args.text_model_name,
        fusion_method=args.fusion_method,
        freeze_video_backbone=True,
        trainable_layers=args.trainable_layers
    ).to(device)

    video = torch.randn(
        args.batch_size,
        args.num_frames,
        3,
        args.img_size,
        args.img_size,
        device=device
    )
    text = ["Normal sample", "Anomalous sample"][:args.batch_size]

    with torch.no_grad():
        logits = model(video, text)

    print(f"Logits shape: {logits.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick multimodal model sanity check')
    parser.add_argument('--video-model-name', default='facebook/timesformer-base-finetuned-k400')
    parser.add_argument('--text-model-name', default='distilbert-base-uncased')
    parser.add_argument('--fusion-method', default='concat')
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-classes', type=int, default=12)
    parser.add_argument('--trainable-layers', type=int, default=2)
    parser.add_argument('--device', default='')
    parser.add_argument('--skip-model', action='store_true')

    main(parser.parse_args())

