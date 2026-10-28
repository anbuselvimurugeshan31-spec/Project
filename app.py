"""
app.py - Streamlit interface.   Run:  streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import (CONFIRMED, PARTIAL, UNVERIFIED, Config, DetectorError,
                      ShelfDetector, annotate, result_rows)

st.set_page_config(page_title="Shelf Product Detector", layout="wide")
st.title("Warehouse Shelf Product Detector")
st.caption("Occlusion-aware: the system only names products it can verify. "
           "Uncertain objects are shown as 'unverified' and are not counted.")


@st.cache_resource(show_spinner="Loading model...")
def load_detector(path: str):
    return ShelfDetector(path)


with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model (pretrained or custom .pt)", "yolov8n.pt")
    imgsz = st.select_slider("Inference size (imgsz)", [640, 960, 1280], value=960)
    use_tiles = st.checkbox("Sliced inference (small products, slower)", value=False)
    conf_confirm = st.slider("Confirm threshold (clear objects)", 0.2, 0.95, 0.50, 0.05)
    conf_partial = st.slider("Partial threshold (occluded objects)", 0.1, 0.9, 0.35, 0.05)
    iou_nms = st.slider("NMS IoU (higher keeps close neighbours)", 0.3, 0.95, 0.70, 0.05)

try:
    detector = load_detector(model_path)
except DetectorError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    class_choice = st.multiselect("Only these classes (empty = all)", sorted(detector.names.values()))

upload = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if upload is None:
    st.info("Upload a shelf photo to begin.")
    st.stop()

img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("That file could not be decoded as an image.")
    st.stop()

cfg = Config(imgsz=imgsz, conf_confirm=conf_confirm, conf_partial=min(conf_partial, conf_confirm),
             iou_nms=iou_nms, use_tiles=use_tiles, allowed_classes=class_choice or None)
try:
    with st.spinner("Detecting..."):
        result = detector.detect(img, cfg)
except DetectorError as e:
    st.error(str(e))
    st.stop()

annotated = annotate(img, result)
c = result.counts
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total identified", result.total_identified)
m2.metric("Confirmed", c[CONFIRMED])
m3.metric("Partially hidden", c[PARTIAL])
m4.metric("Unverified (not counted)", c[UNVERIFIED])
m5.metric("Inference time", f"{result.ms:.0f} ms")

left, right = st.columns([3, 2])
with left:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Green = confirmed, orange = partial, red = unverified", use_container_width=True)
    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("Download annotated image", buf.tobytes(), "annotated.png", "image/png")
with right:
    rows = result_rows(result)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No detections. Try a larger inference size, sliced inference, or a lower partial threshold.")
