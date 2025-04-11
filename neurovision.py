import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile

def draw_label_with_background(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, 
                               text_color=(0, 0, 0), bg_color=(255, 255, 255), padding=3):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness=1)
    cv2.rectangle(img, 
                  (x, y - text_h - padding), 
                  (x + text_w + 2 * padding, y + baseline), 
                  bg_color, 
                  thickness=-1)
    cv2.putText(img, text, (x + padding, y - padding), font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide"
)

# ========== TITLE ==========
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Brain Tumor Detection with YOLOv8</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    return YOLO("./model/best.pt")  # Replace with your path

model = load_model()

# ========== SIDEBAR ==========
st.sidebar.markdown("## 📤 Upload Your MRI")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        results = model.predict(source=tmp.name, conf=0.25, save=False, verbose=False)

    # ========== SIDEBAR PREDICTIONS ==========
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🧪 Prediction Result")
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            st.sidebar.success(f"**{label.capitalize()}**\nConfidence: `{conf:.2f}`")
    else:
        st.sidebar.warning("No tumor detected.")

    # ========== DRAW CUSTOM BOX COLORS ==========
    image_with_boxes = img.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        conf = float(box.conf[0])
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Green or Red
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        draw_label_with_background(image_with_boxes, f"{label.capitalize()} {conf:.2f}", x1, y1)

    prediction_img = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))

    # ========== DISPLAY SIDE-BY-SIDE ==========
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.fromarray(original_img_rgb).resize((214, 214)), caption="📷 Original Image", use_container_width=False)
    with col2:
        st.image(prediction_img.resize((214, 214)), caption="🎯 YOLOv8 Prediction", use_container_width=False)

else:
    st.info("👈 Upload an MRI image to get started.")

# ========== CREDITS ==========
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.9em;'>
        Developed by <b>Mohamed Zakarneh</b> · Powered by <b>YOLOv8</b> · Deployed with <b>Streamlit</b><br>
        🌐 <a href="https://github.com/Zakarneh86/NeuroVision" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)