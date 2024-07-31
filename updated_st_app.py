import streamlit as st
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
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

@st.cache_resource
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((400, 400))  # Resize
    img = np.array(img)[:, :, ::-1]  # Convert to OpenCV format
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@st.cache_resource
def process_image(file_path):
    # Perform inference on the image
    seg_pred_img, mask_img = inference_on_image(file_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, seg_pred_img)
        return tmp_file.name

st.set_page_config(
    layout="centered",
    initial_sidebar_state="collapsed",
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

st.text("")
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
m = buttom_markdown()
load_click = col2.button("Select Image")
m = buttom_markdown()
process_click = col3.button("Run ADR")
m = buttom_markdown()
show_ann_click = col4.button("Show/Hide")

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
        image_zoom(processed_img, mode = "default", zoom_factor=4.0)
        st.markdown("Output Image")
        #image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)
        #st.image(load_image(st.session_state.processed_file_path), caption="Output Image", use_column_width=True)
    else:
        input_img = load_image(st.session_state.selected_file)
        image_zoom(input_img, mode = "default", zoom_factor=4.0)
        st.markdown("Input Image")
        #st.image(load_image(st.session_state.selected_file), caption="Input Image", use_column_width=True)
