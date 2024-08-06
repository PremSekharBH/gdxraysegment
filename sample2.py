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

#@st.cache_resource
def load_image(file_path):
    img = Image.open(file_path)
    #img = img.resize((800, 800))  # Resize
    [m, n, o] = np.shape(img)

    if o == 3:
        img = np.array(img)[:, :, ::-1]
    #img = np.array(img)[:, :, ::-1]  # Convert to OpenCV format
    return img
def load_image_cv2(file_path):
    img = cv2.imread(file_path)
    #img = cv2.resize(img, (800,800))
    return img

#@st.cache_resource
def process_image(file_path):
    # Perform inference on the image
    seg_pred_img, mask_img = inference_on_image(file_path)
    defects_info = extract_defects_info(mask_img)
    st.session_state.defects_info = defects_info
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, seg_pred_img)
        cv2.imwrite('st_out.png', seg_pred_img)
        return tmp_file.name
#@st.cache_resource
def extract_defects_info(mask_img):
    """Extract defect information such as contour and area from the 2D mask image."""
    defects_info = []
    contours, _ = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        print(f"contour detected: {cv2.contourArea(contour)}")
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
def display_defect_info():
    st.sidebar.markdown("## Defect Information")
    if st.session_state.defects_info:
        for i, defect in enumerate(st.session_state.defects_info):
            st.markdown(f"**Defect {i + 1}:**")
            st.text(f"Area: {defect['area']:.2f}")
    if st.session_state.defects_info == []:
        st.sidebar.markdown("No defects detected")
def clear_defect_info():
    st.sidebar.markdown("## Defect Information")
    st.sidebar.markdown("Run ADR to view results")
    # if st.session_state.defects_info:
    #     for i, defect in enumerate(st.session_state.defects_info):
    #         st.markdown(f"**Defect {i + 1}:**")
    #         st.text(f"Area: {defect['area']:.2f}")
    # if st.session_state.defects_info == []:
    #     st.sidebar.markdown("No defects detected")
# with st.sidebar:
#     st.markdown("## Defect Information")
#     #print(f"Defect Info:{st.session_state.defects_info}")
#     if st.session_state.defects_info:
#         for i, defect in enumerate(st.session_state.defects_info):
#             st.markdown(f"**Defect {i + 1}:**")
#             st.text(f"Area: {defect['area']:.2f}")
#     # else:
#     #     st.markdown("No defects detected.")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.text("")
    # Align buttons horizontally
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col1:
        m = buttom_markdown()
        load_click = st.button("Select Image")
    with button_col2:
        m = buttom_markdown()
        process_click = st.button("Run ADR")
    with button_col3:
        m = buttom_markdown()
        show_ann_click = st.button("Show/Hide")

    if load_click:  # Load single Image
        st.session_state.process_done = False
        st.session_state.selected_file = filedialog.askopenfilename(master=root)
        st.session_state.show_annotations = False  # Reset annotation toggle
        st.session_state.defects_info = []
        clear_defect_info()
        if st.session_state.selected_file:
            input_img = load_image_cv2(st.session_state.selected_file)
            image_zoom(input_img, mode="scroll",size=(1024, 1024),zoom_factor=6.0)
            st.markdown("Input Image")
            #st.image(load_image(st.session_state.selected_file), caption="Input Image", use_column_width=True)

    if process_click:
        if st.session_state.selected_file:
            st.session_state.processed_file_path = process_image(st.session_state.selected_file)
            st.session_state.process_done = True
            st.session_state.show_annotations = True
            processed_img = load_image_cv2(st.session_state.processed_file_path)
            cv2.imwrite('st_processout.png', processed_img)
            # process_image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            # process_image_pil = Image.fromarray(process_image_rgb)
            image_zoom(processed_img, mode="scroll", size=(1024, 1024), zoom_factor=6.0)
            st.markdown("Output Image")
            with st.sidebar:
                display_defect_info() # show defect info immediately.
            #st.image(load_image(st.session_state.processed_file_path), caption="Output Image", use_column_width=True)
        else:
            st.error("⚠️ File not selected. Select an image file")

    if show_ann_click and st.session_state.process_done:
        st.session_state.show_annotations = not st.session_state.show_annotations
        with st.sidebar:
                display_defect_info()
        if st.session_state.show_annotations:
            processed_img = load_image(st.session_state.processed_file_path)
            image_zoom(processed_img, mode="scroll", size=(1024, 1024), zoom_factor=6.0)
            st.markdown("Output Image")
        else:
            input_img = load_image(st.session_state.selected_file)
            image_zoom(input_img, mode="scroll", size=(1024, 1024), zoom_factor=6.0)
            st.markdown("Input Image")