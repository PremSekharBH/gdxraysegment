import streamlit as st
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tempfile
from maskrcnn_infer import inference_on_image
from st_utils import buttom_markdown
from streamlit_image_zoom import image_zoom

# Default settings
default_detector_settings = {
    "UT_ADR": {
        "dect_conf": 0.3,
        "nms_conf": 0.4,
    }
}

# Initialize session state
if 'process_done' not in st.session_state:
    st.session_state.process_done = False
if 'show_annotations' not in st.session_state:
    st.session_state.show_annotations = False
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'processed_file_path' not in st.session_state:
    st.session_state.processed_file_path = None
if 'defects_info' not in st.session_state:
    st.session_state.defects_info = []

@st.cache_resource
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((400, 400))  # Resize
    img = np.array(img)[:, :, ::-1]  # Convert to OpenCV format
    return img

@st.cache_resource
def process_image(file_path):
    # Perform inference on the image
    seg_pred_img, mask_img = inference_on_image(file_path)
    defects_info = extract_defects_info(mask_img)
    st.session_state.defects_info = defects_info
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, seg_pred_img)
        return tmp_file.name

def extract_defects_info(mask_img):
    """Extract defect information such as contour and area from the 2D mask image."""
    defects_info = []
    contours, _ = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        defects_info.append({"contour": contour, "area": area})
    return defects_info

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👓",
    page_title="Weld ADR Segmentation",
)

img = Image.open("WayGateLogo.png").resize((500 * 2, 149 * 2))
st.image(img, use_column_width=True)
st.markdown(
    "<h1 style='text-align: center; color: DarkSeaGreen; font-size: 50px'> Weld ADR Segmentation </h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)

# Left sidebar for defect information
with st.sidebar:
    st.markdown("## Defect Information")
    if st.session_state.defects_info:
        for i, defect in enumerate(st.session_state.defects_info):
            st.markdown(f"**Defect {i + 1}:**")
            st.text(f"Area: {defect['area']:.2f}")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.text("")
    # Align buttons horizontally
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col1:
        load_click = st.button("Select Image")
    with button_col2:
        process_click = st.button("Run ADR")
    with button_col3:
        show_ann_click = st.button("Show/Hide")

    if load_click:  # Load single Image
        st.session_state.process_done = False
        st.session_state.selected_file = filedialog.askopenfilename(master=root)
        st.session_state.show_annotations = False  # Reset annotation toggle

        if st.session_state.selected_file:
            st.image(load_image(st.session_state.selected_file), caption="Input Image", use_column_width=True)

    if process_click:
        if st.session_state.selected_file:
            st.session_state.processed_file_path = process_image(st.session_state.selected_file)
            st.session_state.process_done = True
            st.session_state.show_annotations = True
            st.image(load_image(st.session_state.processed_file_path), caption="Output Image", use_column_width=True)
        else:
            st.error("⚠️ File not selected. Select an image file")

    if show_ann_click and st.session_state.process_done:
        st.session_state.show_annotations = not st.session_state.show_annotations

        if st.session_state.show_annotations:
            processed_img = load_image(st.session_state.processed_file_path)
            image_zoom(processed_img, mode="default", zoom_factor=4.0)
            st.markdown("Output Image")
        else:
            input_img = load_image(st.session_state.selected_file)
            image_zoom(input_img, mode="default", zoom_factor=4.0)
            st.markdown("Input Image")
