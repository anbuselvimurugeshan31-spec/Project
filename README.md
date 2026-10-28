# Occlusion-Aware Warehouse Shelf Product Detector

Detects and locates products on a shelf (YOLO + OpenCV), even when products touch or partly hide each other,
and **refuses to name objects it cannot verify**.

## 1. Recommended approach

| Choice | Why it fits |
|---|---|
| Ultralytics YOLOv8n (pretrained COCO) | Runs on a student laptop CPU; one pass gives one box per object, so touching products stay separate. |
| High NMS IoU (0.70) | Neighbouring products legitimately overlap in the image; a low threshold would delete real products. |
| `imgsz=960` (+ optional sliced inference) | Small products shrink to a few pixels at the default 640. |
| **Three-status output (design constraint)** | A box alone cannot prove a product is identified. Only CONFIRMED and PARTIAL objects are named and counted. |
| Custom training (optional) | COCO has 80 generic classes. It cannot know your brands/SKUs. |

### The design constraint
> The system may only assert what it can verify. A detection is **CONFIRMED** (confident, mostly visible),
> **PARTIAL** (identified but hidden/cut off - lower confidence bar, flagged), or **UNVERIFIED**
> (low confidence or >=80% hidden - drawn as "unverified object", no product name, **not counted**).

| Status | Rule (defaults, all adjustable in `Config`) | Named? | Counted? |
|---|---|---|---|
| CONFIRMED | conf >= 0.50 and < 30% hidden and not touching image edge | yes | yes |
| PARTIAL | conf >= 0.35 and (>= 30% hidden or touching edge) and < 80% hidden | yes, "(partial)" | yes, separately |
| UNVERIFIED | anything else (conf < 0.35, mostly hidden, or unoccluded but conf < 0.50) | **no** | **no** |

How "hidden" is estimated: the fraction of a box covered by other detections that are *more confident*
(assumed to be in front). Boxes carry no depth information, so this is an estimate, not a measurement.
**Instance segmentation** (`yolov8n-seg.pt`) would give exact visible regions and is the better choice for heavy
overlap; it is a good "future work" item. Boxes were chosen here for simplicity and CPU speed.

## 2. Architecture

```
 Image / video / webcam
          |
   Pre-processing (load, validate, resize to imgsz)
          |
   YOLO inference (conf floor 0.15)  --optional-->  tiled passes + class-wise NMS merge
          |
   NMS (IoU 0.70) -> raw boxes + class guesses + confidence
          |
   Occlusion estimator (box coverage by more-confident neighbours, edge truncation)
          |
   Verification gate -> CONFIRMED | PARTIAL | UNVERIFIED
          |
   Annotator (colour-coded boxes, IDs, counts) + table
          |
   CLI (main.py) / Streamlit (app.py) / saved image or video
```

## 3. Install
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
First run downloads `yolov8n.pt` automatically (needs internet once).

## 4. Files
```
shelf_detector/
  detector.py      core logic (detection, occlusion, verification gate, drawing)
  main.py          command-line: image / video / webcam
  app.py           Streamlit UI
  test_logic.py    model-free tests of the decision logic
  nms_demo.py      IoU + NMS teaching demo
  train.py         optional custom training
  evaluate.py      precision / recall / mAP on your labelled data
  data.yaml        dataset config example
  requirements.txt
```

## 5. Run
```bash
python test_logic.py                              # no model needed
python nms_demo.py                                # IoU / NMS explanation
python main.py --image shelf.jpg                  # saves output.jpg, opens a window
python main.py --image shelf.jpg --tiles --imgsz 1280   # small products
python main.py --image shelf.jpg --classes bottle cup   # restrict to relevant classes
python main.py --video shelf.mp4 --out result.mp4
python main.py --webcam
streamlit run app.py
```
Useful knobs: `--iou` (NMS), `--conf-confirm`, `--conf-partial`, `--imgsz`, `--tiles`.

## 6. Sample input and expected output
**Input:** a photo of a shelf with 5-10 objects COCO knows (bottles, cups, books, bowls) packed tightly, some
partly behind others. Take it yourself, straight-on, good light.
**Expected (not guaranteed):** coloured boxes with `#id label conf`, a banner `Confirmed / Partial / Unverified`,
and a console table. Front objects green, partly hidden ones orange, dubious ones red "unverified object".
Exact counts depend on your image; nothing here has been measured on a real shelf photo yet.

## 7. Optional custom training
1. **Collect**: 200-500+ shelf photos from real angles, lighting, full and half-empty shelves, products touching/overlapping.
2. **Label** with Roboflow, CVAT or LabelImg (export "YOLO"). One box per product. **For hidden products, box only the
   visible part, consistently.** Include some images with no target products (hard negatives).
3. **Structure** (matches `data.yaml`):
```
datasets/warehouse/
  images/train  images/val  images/test
  labels/train  labels/val  labels/test      # each label file: class x_center y_center w h  (0-1)
```
4. Split about 70/20/10 by *shelf/session*, not randomly per photo, so test images are truly unseen.
5. **Train**: `python train.py --data data.yaml --epochs 100 --imgsz 960` (use Colab/Kaggle GPU if possible).
6. **Use**: `python main.py --image unseen.jpg --model runs/shelf_custom/weights/best.pt`
7. **Evaluate**: `python evaluate.py --data data.yaml --model runs/shelf_custom/weights/best.pt --split test`

When is it required? Whenever you need brand/SKU names, packaging COCO lacks (cartons, sacks, tins), or
reliable counts of identical-looking items.

## 8. Demonstration plan (for your lecturer)
1. Show the problem: a crowded shelf photo with some partly hidden items.
2. Run `python test_logic.py` - proves the verification rules work (all 9 pass).
3. Run `python nms_demo.py` - explain IoU and why NMS threshold matters.
4. Run the Streamlit app on your photo; show green/orange/red boxes and the table.
5. Move the sliders live: raise the confirm threshold -> objects drop to "unverified"; lower NMS IoU -> neighbours merge.
6. Be upfront: pretrained COCO names only generic classes; show where it labels wrongly or abstains, then explain custom training.

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| `Ultralytics is not installed` | `pip install ultralytics` inside the activated venv |
| Weights fail to download | Check internet/proxy; or download `yolov8n.pt` manually into the project folder |
| `Image not found` | Check the path/quotes; use full path |
| No window opens | Headless/remote machine: add `--no-show`; output still saved |
| Everything "unverified" | Objects are not COCO classes (expected) -> use custom training; or lower `--conf-confirm` knowingly |
| Neighbours merged into one box | Raise `--iou` (0.7-0.8), raise `--imgsz`, try `--tiles` |
| Duplicate boxes on one product | Lower `--iou` (0.5-0.6) |
| Slow on CPU | Use `--imgsz 640`, `--skip 4` for video, avoid `--tiles` |
| Webcam not opening | Try `--camera-index 1`; close other apps using the camera |
| Streamlit `use_container_width` warning | Harmless on newer versions |
