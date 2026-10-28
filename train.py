"""
train.py - OPTIONAL: fine-tune YOLO on your own warehouse dataset.

    python train.py --data data.yaml --epochs 100
    python train.py --data data.yaml --model yolov8n.pt --imgsz 960 --batch 4 --device cpu

On CPU training is slow; for real training use free Google Colab / Kaggle GPUs, then copy
weights/best.pt back to your laptop for inference.
"""
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="starting weights (transfer learning)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu", help="'cpu', '0' for first GPU")
    p.add_argument("--name", default="shelf_custom")
    a = p.parse_args()

    if not Path(a.data).exists():
        sys.exit(f"[ERROR] Dataset config not found: {a.data}")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] Install first:  pip install ultralytics")

    try:
        model = YOLO(a.model)
        model.train(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project="runs", name=a.name, patience=30, workers=2,
            mosaic=1.0,          # mosaic mixes 4 images -> objects appear partly cut/hidden
            degrees=5.0,         # small rotations (tilted packages)
            scale=0.5, fliplr=0.5,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # lighting / colour variation
        )
    except Exception as e:
        sys.exit(f"[ERROR] Training failed: {e}")
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Use them:  python main.py --image shelf.jpg --model {best}")


if __name__ == "__main__":
    main()
