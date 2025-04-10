import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO("./model/best.pt")  # Replace with your model path

model = load_model()



# ========== TITLE ==========
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Brain Tumor Detection with YOLOv8</h1>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.markdown("## 📤 Upload Your MRI")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save temporarily for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        results = model.predict(source=tmp.name, conf=0.25, save=False, verbose=False)

    '''# Annotated prediction image
    prediction_img = results[0].plot()
    prediction_img_rgb = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)'''
    prediction_img = Image.fromarray(results[0].plot())

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

    # ========== DISPLAY SIDE-BY-SIDE ==========
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img_rgb, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(prediction_img, caption="🎯 YOLOv8 Prediction", use_column_width=True)

else:
    st.info("👈 Upload an MRI image to get started.")

# ========== CREDITS ==========
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.9em;'>
        Developed by <b>You</b> · Powered by <b>YOLOv8</b> · Deployed with <b>Streamlit</b><br>
        🌐 <a href="https://github.com/your-repo" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)