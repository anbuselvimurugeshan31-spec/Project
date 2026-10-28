"""
main.py - command-line interface.

Examples
    python main.py --image shelf.jpg
    python main.py --image shelf.jpg --tiles --imgsz 1280
    python main.py --video shelf.mp4 --out result.mp4 --skip 2
    python main.py --webcam
    python main.py --image shelf.jpg --model runs/shelf_custom/weights/best.pt
"""
import argparse
import sys

import cv2

from detector import (CONFIRMED, PARTIAL, UNVERIFIED, Config, DetectorError,
                      ShelfDetector, annotate, load_image)


def build_cfg(a) -> Config:
    return Config(imgsz=a.imgsz, conf_confirm=a.conf_confirm, conf_partial=a.conf_partial,
                  iou_nms=a.iou, use_tiles=a.tiles, allowed_classes=a.classes, device=a.device)


def print_report(result):
    c = result.counts
    print(f"\nConfirmed: {c[CONFIRMED]} | Partially hidden: {c[PARTIAL]} | "
          f"Unverified (NOT counted): {c[UNVERIFIED]} | Total identified: {result.total_identified} "
          f"| Inference: {result.ms:.0f} ms\n")
    print(f"{'ID':>3}  {'STATUS':<11} {'LABEL':<18} {'CONF':>5}  {'BOX (x1, y1, x2, y2)':<26} {'HIDDEN':>6}  REASON")
    for d in result.detections:
        label = "(not identified)" if d.status == UNVERIFIED else d.label
        print(f"{d.id:>3}  {d.status:<11} {label:<18} {d.conf:>5.2f}  {str(d.box):<26} {d.occlusion:>5.0%}  {d.reason}")
    if not result.detections:
        print("No detections. Try lowering --conf-partial, raising --imgsz, or using --tiles.")


def show(window, img) -> bool:
    """Show an image; returns False if no display is available."""
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, img)
        return True
    except cv2.error:
        print("(No display available - skipping window; output is still saved to file.)")
        return False


def run_image(detector, cfg, a):
    img = load_image(a.image)
    result = detector.detect(img, cfg)
    print_report(result)
    out = annotate(img, result)
    if not cv2.imwrite(a.out, out):
        raise DetectorError(f"Could not write output image to {a.out}")
    print(f"\nSaved annotated image -> {a.out}")
    if not a.no_show and show("Shelf detections (press any key to close)", out):
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(detector, cfg, a, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise DetectorError(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    writer, idx, result = None, 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if result is None or idx % (a.skip + 1) == 0:
                result = detector.detect(frame, cfg)
            vis = annotate(frame, result)
            if a.out and writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if writer is not None:
                writer.write(vis)
            if not a.no_show:
                if not show("Shelf video (press q to quit)", vis):
                    a.no_show = True
                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video -> {a.out}")
        cv2.destroyAllWindows()
    if result is not None:
        print_report(result)


def main():
    p = argparse.ArgumentParser(description="Occlusion-aware shelf product detector")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    src.add_argument("--webcam", action="store_true")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--out", default=None, help="output file (default output.jpg for images)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    p.add_argument("--conf-confirm", type=float, default=0.50)
    p.add_argument("--conf-partial", type=float, default=0.35)
    p.add_argument("--tiles", action="store_true", help="sliced inference for small products")
    p.add_argument("--classes", nargs="+", default=None, help="only keep these class names")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip", type=int, default=2, help="video: frames to skip between detections")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    try:
        detector = ShelfDetector(a.model)
        cfg = build_cfg(a)
        if a.image:
            a.out = a.out or "output.jpg"
            run_image(detector, cfg, a)
        elif a.video:
            run_video(detector, cfg, a, a.video)
        else:
            run_video(detector, cfg, a, a.camera_index)
    except DetectorError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
