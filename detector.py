"""
detector.py - Occlusion-aware shelf product detector.

Core idea (the project's design constraint):
    The system only ASSERTS what it can verify.
    Every detection gets a status:
      CONFIRMED  - confident and not significantly hidden      -> counted
      PARTIAL    - identified, but hidden by a neighbour or cut
                   by the image edge (lower bar, flagged)        -> counted separately
      UNVERIFIED - low confidence or mostly hidden; NO product
                   name is claimed                               -> NOT counted
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CONFIRMED = "CONFIRMED"
PARTIAL = "PARTIAL"
UNVERIFIED = "UNVERIFIED"

# BGR colours used when drawing
COLORS = {
    CONFIRMED: (0, 180, 0),      # green
    PARTIAL: (0, 140, 255),      # orange
    UNVERIFIED: (80, 80, 220),   # red-ish
}


class DetectorError(Exception):
    """Friendly, user-readable error."""


@dataclass
class Config:
    imgsz: int = 960              # larger than default 640 -> better for small products
    conf_floor: float = 0.15      # model-level floor: below this a box is discarded entirely
    conf_confirm: float = 0.50    # needed to be CONFIRMED (unoccluded object)
    conf_partial: float = 0.35    # lower bar for objects that are visibly occluded/truncated
    occ_partial: float = 0.30     # >= this fraction hidden -> treated as occluded
    occ_hidden: float = 0.80      # >= this fraction hidden -> too little visible, UNVERIFIED
    iou_nms: float = 0.70         # NMS IoU threshold; HIGH keeps touching neighbours separate
    max_det: int = 300
    edge_margin: int = 3          # pixels; box this close to the image border = truncated
    min_box_px: int = 12          # ignore boxes smaller than this (noise)
    use_tiles: bool = False       # sliced inference for very small products
    tile_size: int = 640
    tile_overlap: float = 0.25
    allowed_classes: Optional[List[str]] = None  # e.g. ["bottle", "cup"]; None = all
    device: str = "cpu"


@dataclass
class Detection:
    id: int
    label: str                    # model's class guess
    conf: float
    box: Tuple[int, int, int, int]
    occlusion: float              # 0..1 estimated hidden fraction of the box
    truncated: bool               # touches the image border
    status: str
    reason: str

    @property
    def shown_label(self) -> str:
        """Name we are willing to claim. Unverified objects get no product name."""
        return "unverified object" if self.status == UNVERIFIED else self.label


@dataclass
class Result:
    detections: List[Detection]
    ms: float

    @property
    def counts(self) -> Dict[str, int]:
        c = {CONFIRMED: 0, PARTIAL: 0, UNVERIFIED: 0}
        for d in self.detections:
            c[d.status] += 1
        return c

    @property
    def total_identified(self) -> int:
        c = self.counts
        return c[CONFIRMED] + c[PARTIAL]


# --------------------------------------------------------------------------- #
# Geometry + decision logic (pure functions: testable without a model)
# --------------------------------------------------------------------------- #
def iou(a, b) -> float:
    """Intersection over Union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def occlusion_fraction(i: int, boxes, confs) -> float:
    """
    Estimate how much of box i is covered by OTHER detections.

    Heuristic: a neighbour is assumed to be IN FRONT of box i if its confidence is
    higher (a less-hidden object is usually detected more confidently). Boxes cannot
    tell us true depth, so this is an estimate; instance segmentation would be exact.
    """
    x1, y1, x2, y2 = boxes[i]
    w, h = max(int(round(x2 - x1)), 1), max(int(round(y2 - y1)), 1)
    covered = np.zeros((h, w), dtype=bool)
    for j, b in enumerate(boxes):
        if j == i:
            continue
        if confs[j] < confs[i] or (confs[j] == confs[i] and j > i):
            continue  # not considered "in front"
        ix1, iy1 = max(x1, b[0]), max(y1, b[1])
        ix2, iy2 = min(x2, b[2]), min(y2, b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        covered[int(iy1 - y1):int(round(iy2 - y1)), int(ix1 - x1):int(round(ix2 - x1))] = True
    return float(covered.mean())


def assess(boxes, confs, labels, img_w: int, img_h: int, cfg: Config) -> List[Detection]:
    """Turn raw model output into CONFIRMED / PARTIAL / UNVERIFIED detections."""
    out: List[Detection] = []
    order = sorted(range(len(boxes)), key=lambda k: -confs[k])  # IDs: most confident first
    for new_id, i in enumerate(order, start=1):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        conf = float(confs[i])
        occ = occlusion_fraction(i, boxes, confs)
        m = cfg.edge_margin
        truncated = x1 <= m or y1 <= m or x2 >= img_w - m or y2 >= img_h - m
        occluded = occ >= cfg.occ_partial or truncated

        if conf < cfg.conf_partial:
            status, reason = UNVERIFIED, f"confidence {conf:.2f} below {cfg.conf_partial}"
        elif occ >= cfg.occ_hidden:
            status, reason = UNVERIFIED, f"{occ:.0%} hidden - too little visible to identify"
        elif occluded:
            why = "cut by image edge" if truncated and occ < cfg.occ_partial else f"{occ:.0%} hidden"
            status, reason = PARTIAL, f"identified but partially visible ({why})"
        elif conf >= cfg.conf_confirm:
            status, reason = CONFIRMED, "clearly visible and confident"
        else:
            status, reason = UNVERIFIED, f"confidence {conf:.2f} below {cfg.conf_confirm} for an unoccluded object"

        out.append(Detection(
            id=new_id, label=str(labels[i]), conf=conf,
            box=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            occlusion=occ, truncated=truncated, status=status, reason=reason,
        ))
    return out


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class ShelfDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorError("Ultralytics is not installed. Run:  pip install ultralytics") from e
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise DetectorError(
                f"Could not load model '{model_path}'. On the first run the pretrained weights "
                f"are downloaded automatically, so check your internet connection. "
                f"For custom weights check the file path. Details: {e}"
            ) from e
        self.model_path = model_path
        self.names = self.model.names  # {class_id: name}

    def _predict(self, img, imgsz: int, cfg: Config):
        res = self.model.predict(
            img, imgsz=imgsz, conf=cfg.conf_floor, iou=cfg.iou_nms,
            max_det=cfg.max_det, device=cfg.device, verbose=False,
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    def _predict_tiled(self, img, cfg: Config):
        """Full-image pass + overlapping tiles, merged with class-wise NMS."""
        import torch
        import torchvision

        h, w = img.shape[:2]
        ts = cfg.tile_size
        step = max(int(ts * (1 - cfg.tile_overlap)), 1)

        def starts(n):
            if n <= ts:
                return [0]
            s = list(range(0, n - ts + 1, step))
            if s[-1] != n - ts:
                s.append(n - ts)
            return s

        all_b, all_c, all_k = [], [], []
        b, c, k = self._predict(img, cfg.imgsz, cfg)
        all_b.append(b); all_c.append(c); all_k.append(k)
        for y0 in starts(h):
            for x0 in starts(w):
                crop = img[y0:y0 + ts, x0:x0 + ts]
                b, c, k = self._predict(crop, ts, cfg)
                if len(b):
                    b = b + np.array([x0, y0, x0, y0], dtype=np.float32)
                all_b.append(b); all_c.append(c); all_k.append(k)
        b = np.concatenate(all_b); c = np.concatenate(all_c); k = np.concatenate(all_k)
        if len(b) == 0:
            return b, c, k
        keep = torchvision.ops.batched_nms(
            torch.tensor(b, dtype=torch.float32), torch.tensor(c, dtype=torch.float32),
            torch.tensor(k), cfg.iou_nms,
        ).numpy()
        return b[keep], c[keep], k[keep]

    def detect(self, img_bgr: np.ndarray, cfg: Optional[Config] = None) -> Result:
        cfg = cfg or Config()
        if img_bgr is None or img_bgr.size == 0:
            raise DetectorError("Empty image passed to the detector.")
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()
        try:
            boxes, confs, cls = (self._predict_tiled(img_bgr, cfg) if cfg.use_tiles
                                 else self._predict(img_bgr, cfg.imgsz, cfg))
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(f"Inference failed: {e}") from e
        ms = (time.perf_counter() - t0) * 1000

        labels = [self.names[int(k)] for k in cls]
        keep = []
        allowed = {a.lower() for a in cfg.allowed_classes} if cfg.allowed_classes else None
        for i, b in enumerate(boxes):
            if min(b[2] - b[0], b[3] - b[1]) < cfg.min_box_px:
                continue
            if allowed and labels[i].lower() not in allowed:
                continue
            keep.append(i)
        boxes = [boxes[i] for i in keep]
        confs = [float(confs[i]) for i in keep]
        labels = [labels[i] for i in keep]
        return Result(assess(boxes, confs, labels, w, h, cfg), ms)


# --------------------------------------------------------------------------- #
# Drawing / IO helpers
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise DetectorError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise DetectorError(f"File exists but is not a readable image: {path}")
    return img


def annotate(img: np.ndarray, result: Result) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 500))
    scale = max(0.5, max(h, w) / 1600)
    for d in result.detections:
        x1, y1, x2, y2 = d.box
        col = COLORS[d.status]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick if d.status != UNVERIFIED else max(1, thick - 1))
        text = f"#{d.id} {d.shown_label} {d.conf:.2f}" + (" (partial)" if d.status == PARTIAL else "")
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y1 + th + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base), col, -1)
        cv2.putText(out, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    c = result.counts
    banner = (f"Confirmed: {c[CONFIRMED]}  Partial: {c[PARTIAL]}  "
              f"Unverified (not counted): {c[UNVERIFIED]}")
    (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + base + 14), (0, 0, 0), -1)
    cv2.putText(out, banner, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def result_rows(result: Result) -> List[dict]:
    rows = []
    for d in result.detections:
        x1, y1, x2, y2 = d.box
        rows.append({
            "ID": d.id, "Status": d.status,
            "Label": "(not identified)" if d.status == UNVERIFIED else d.label,
            "Model guess": d.label, "Confidence": round(d.conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Hidden %": round(d.occlusion * 100), "Reason": d.reason,
        })
    return rows
