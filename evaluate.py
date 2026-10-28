"""
evaluate.py - measure precision, recall, mAP and speed on a labelled validation/test set.
Only run this after you have a labelled dataset; report ONLY the numbers it prints.

    python evaluate.py --data data.yaml --model runs/shelf_custom/weights/best.pt --split test
"""
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    try:
        from ultralytics import YOLO
        m = YOLO(a.model).val(data=a.data, imgsz=a.imgsz, split=a.split,
                              device=a.device, iou=0.6, verbose=False)
    except ImportError:
        sys.exit("[ERROR] pip install ultralytics")
    except Exception as e:
        sys.exit(f"[ERROR] Evaluation failed: {e}")
    print(f"Precision : {m.box.mp:.3f}")
    print(f"Recall    : {m.box.mr:.3f}")
    print(f"mAP@0.50  : {m.box.map50:.3f}")
    print(f"mAP@0.50:0.95 : {m.box.map:.3f}")
    print("Speed (ms/img):", {k: round(v, 1) for k, v in m.speed.items()})


if __name__ == "__main__":
    main()
