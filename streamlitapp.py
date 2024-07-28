import streamlit as st
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from yolox_inference import InferYoloX
from maskrcnn_infer import inference_on_image
import matplotlib.pyplot as plt
import cv2
from config import _CLASSES, model_path
import glob
from st_utils import buttom_markdown
import os
import numpy as np
import time

use_case = "UT_ADR"
default_detector_settings = {}
default_detector_settings["UT_ADR"] = {
    "dect_conf": 0.3,
    "nms_conf": 0.4,
}

if 'counter' not in st.session_state: 
    st.session_state.counter = 0
    
if 'process_click' not in st.session_state: 
    st.session_state.process_click = False
if 'process_done' not in st.session_state: 
    st.session_state.process_done = False

if 'show_ann_click' not in st.session_state:
    st.session_state.show_ann_click = False
if 'hide_ann_click' not in st.session_state:
    st.session_state.hide_ann_click = False
    
if 'load_click' not in st.session_state: 
    st.session_state.load_click = False

if 'load_folder' not in st.session_state: 
    st.session_state.load_folder = False

if 'selected_file' not in st.session_state: 
    st.session_state.selected_file = False
    
    
print('-----'*4)
for key in st.session_state:
    print(f'{key} : {st.session_state[key]}')
    
@st.cache_resource
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((400, 400)) #resize
    img = np.array(img)[:, :, ::-1]  
    return img


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
col1,col2,col3,col4,col5 = st.columns([1,1,1,1,1])

m = buttom_markdown()
load_click = col2.button("Select Image")
# m = buttom_markdown()
# load_folder = col3.button("Select Folder")
m = buttom_markdown()
process_click = col3.button("Run ADR")
m = buttom_markdown()
show_ann_click = col4.button("Show/Hide")

if load_click: # Load single Image
    st.session_state.process_click = False
    st.session_state.process_done = False
    st.session_state.load_click = True

    #st.session_state.load_folder = False
    st.session_state.fname = st.text_input(
        "Selected Image File:", filedialog.askopenfilename(master=root)
    )
    fig, ax1 =plt.subplots(figsize = (36, 36))
    #img = Image.open(st.session_state.fname)
    input_img = load_image(st.session_state.fname)
    start_time = time.time()
    #ax1.imshow(np.array(img)[:, :, ::-1])
    ax1.imshow(input_img)
    #ax1.imshow(cv2.imread(st.session_state.fname)[:, :, ::-1])
    ax1.axis("off")
    ax1.set_title("Input Image")

    st.pyplot(fig)
    print(f"time taken to disply image:{start_time - time.time()}")
    if len(st.session_state.fname) == 0:
        st.error("⚠️ File not selected. Select a image file")
        st.stop()
    

def showPhoto(photo):
    col2.image(cv2.imread(photo))
    col1.write(f"Index as a session_state attribute: {st.session_state.counter}")
    
    ## Increments the counter to get next photo
    st.session_state.counter += 1
    if st.session_state.counter >= len(st.session_state.folder_image_files):
        st.session_state.counter = 0
    
    st.session_state.load_folder == True
        
if process_click:
    st.session_state.process_click = True
    print(f'Select Single Button Clicked : {st.session_state.load_click}')
    #print(f'Select Folder Button Clicked : {st.session_state.load_folder}')
    
    #if not ((st.session_state.load_click == True) or (st.session_state.load_folder == True)):
    if not (st.session_state.load_click == True):
        st.error("⚠️ File / Folder not selected. Select a image / folder using 'Select Single Image' / 'Select Folder' button")
        st.stop()

if show_ann_click:
    st.session_state.show_ann_click = True
    print(f'Show Annotation Button Clicked : {st.session_state.show_ann_click}')

    if not (st.session_state.load_click == True):
        st.error("⚠️ File / Folder not selected. Select a image / folder using 'Select Single Image' / 'Select Folder' button")
        st.stop()

    if not (st.session_state.process_done == True):
        st.error("⚠️ Process not completed. Select 'Process' button")
        st.stop()
        


if (st.session_state.load_click == True) and (st.session_state.process_click==True):
    seg_pred_img = inference_on_image(st.session_state.fname)
    fig, ax2 =plt.subplots(figsize = (36, 36))

    cv2.imwrite("output.jpg", seg_pred_img)

    ax2.imshow(seg_pred_img[:, :, ::-1])
    ax2.axis("off")
    ax2.set_title("Output Image")
    
    st.pyplot(fig)
    st.session_state.process_done = True
    st.session_state.process_click= False


if (st.session_state.show_ann_click==True) and (st.session_state.load_click == True) and (st.session_state.process_done == True):
    st.session_state.hide_ann_click = not st.session_state.hide_ann_click
    if (st.session_state.hide_ann_click == True):
        fig, ax1 =plt.subplots(figsize = (36, 36))
        #ax1.imshow(input_img)
        ax1.imshow(load_image(st.session_state.fname))
        #ax1.imshow(cv2.imread(st.session_state.fname)[:, :, ::-1])
        ax1.axis("off")
        ax1.set_title("Input Image")

        st.pyplot(fig)
    if not (st.session_state.hide_ann_click == True):
        fig, ax1 =plt.subplots(figsize = (36, 36))
        out_img = load_image("output.jpg")
        ax1.imshow(out_img)
        #ax1.imshow(cv2.imread("output.tif")[:, :, ::-1])
        ax1.axis("off")
        ax1.set_title("Output Image")

        st.pyplot(fig)
    st.session_state.show_ann_click = not st.session_state.show_ann_click

        
