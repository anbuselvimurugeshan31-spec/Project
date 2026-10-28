"""
test_logic.py - tests the decision logic WITHOUT needing a model or internet.
Run:  python test_logic.py
"""
from detector import (CONFIRMED, PARTIAL, UNVERIFIED, Config, assess, iou,
                      occlusion_fraction)

cfg = Config()
W, H = 640, 480


def run(boxes, confs):
    return assess(boxes, confs, ["bottle"] * len(boxes), W, H, cfg)


def by_conf(dets, conf):
    return next(d for d in dets if abs(d.conf - conf) < 1e-9)


# IoU
assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9

# 1. clear, confident object -> CONFIRMED
assert run([(100, 100, 200, 200)], [0.80])[0].status == CONFIRMED

# 2. unoccluded but only moderately confident -> UNVERIFIED (abstain)
assert run([(100, 100, 200, 200)], [0.40])[0].status == UNVERIFIED

# 3. very low confidence -> UNVERIFIED
assert run([(100, 100, 200, 200)], [0.20])[0].status == UNVERIFIED

# 4. neighbour partly hides a product: front one CONFIRMED, hidden one PARTIAL
d = run([(100, 100, 200, 200), (150, 100, 250, 200)], [0.90, 0.50])
assert by_conf(d, 0.90).status == CONFIRMED
assert by_conf(d, 0.50).status == PARTIAL
assert abs(by_conf(d, 0.50).occlusion - 0.5) < 1e-6

# 5. almost completely covered -> UNVERIFIED even with 0.40 confidence
d = run([(100, 100, 200, 200), (105, 105, 195, 195)], [0.90, 0.45])
assert by_conf(d, 0.45).status == UNVERIFIED

# 6. cut off by the image edge -> PARTIAL (not CONFIRMED)
assert run([(0, 100, 80, 200)], [0.70])[0].status == PARTIAL

# 7. touching but NOT overlapping neighbours stay independent, both CONFIRMED
d = run([(100, 100, 200, 200), (200, 100, 300, 200)], [0.85, 0.80])
assert all(x.status == CONFIRMED for x in d)

# 8. no detections
assert run([], []) == []

# 9. unverified detections never expose a product name
d = run([(100, 100, 200, 200)], [0.20])[0]
assert d.shown_label == "unverified object"

print("All 9 logic tests passed.")
