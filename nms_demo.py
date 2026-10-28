"""
nms_demo.py - teaching script: IoU and Non-Maximum Suppression on made-up boxes.
Run:  python nms_demo.py
"""
from detector import iou


def nms(boxes, scores, iou_thr):
    """Greedy NMS: keep the best box, drop others that overlap it by more than iou_thr."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) <= iou_thr]
    return keep


# Three products side by side. Boxes 0 and 1 overlap a little (two DIFFERENT neighbours).
# Box 2 is a duplicate detection of box 0 (same product detected twice).
boxes = [(100, 100, 200, 300), (140, 100, 240, 300), (102, 102, 202, 302)]
scores = [0.90, 0.85, 0.60]

print("IoU(0,1) neighbours  :", round(iou(boxes[0], boxes[1]), 3))
print("IoU(0,2) duplicate   :", round(iou(boxes[0], boxes[2]), 3))
for thr in (0.30, 0.50, 0.70, 0.95):
    k = nms(boxes, scores, thr)
    print(f"NMS IoU={thr:.2f} -> keeps boxes {k}  ({len(k)} products)")
print("\nCorrect answer is 2 products (boxes 0 and 1). A too-LOW threshold (0.30) deletes the real "
      "neighbour because it overlaps; a too-HIGH threshold (0.95) keeps the duplicate.\n"
      "The right value lies between the neighbour IoU and the duplicate IoU; tune it on your own shelf images.")
