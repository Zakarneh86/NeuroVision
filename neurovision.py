import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Load your trained model
@st.cache_resource
def load_model():
    return YOLO("./model/best.pt")

model = load_model()

st.title("🧠 Brain Tumor Detection with YOLOv8")

# Sidebar: Upload and results
st.sidebar.header("Upload Image & View Result")
uploaded_file = st.sidebar.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read original image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        results = model.predict(source=tmp.name, conf=0.25, save=False, verbose=False)

    # Get prediction image (with bounding boxes)
    prediction_img = results[0].plot()
    prediction_img_rgb = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)

    # Sidebar: show prediction info
    st.sidebar.markdown("### 🧪 Prediction Result")
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            st.sidebar.write(f"**Class**: {label}")
            st.sidebar.write(f"**Confidence**: {conf:.2f}")
    else:
        st.sidebar.warning("No tumor detected.")

    # Main: Side-by-side image display
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img_rgb, caption="🧾 Original Image", use_column_width=True)
    with col2:
        st.image(prediction_img_rgb, caption="🎯 YOLO Prediction", use_column_width=True)