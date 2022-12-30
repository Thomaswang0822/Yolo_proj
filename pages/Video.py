import time
from io import BytesIO, BufferedReader

import cv2
import numpy as np
import streamlit as st
from PIL import Image
# Import prediction script
from prediction import Yolo_Predictor
import tempfile



# Set page configurations
# noinspection PyTypeChecker
st.set_page_config(
    page_title="Video Detection",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Object Detection in a Video")

cur_video = st.container()

# Current Input Image Displayer
cur_video.subheader("Your input image:")
cur_video.vid_display = st.empty()
cur_video.vid_warning = st.empty()
cur_video.vid_warning.warning("No Video")

file_upload, detection = st.tabs(["Upload Video", "Object detection"])

# file_upload tab
file_upload.header("Upload Image")
file_upload.subheader("Please upload your image")
file_video = file_upload.file_uploader("One Video in .mp4 or .mov",
                                       type=['mp4', 'mov'],
                                       accept_multiple_files=False,
                                       label_visibility="visible")
video = file_video
if file_video is not None:
    cur_video.vid_display.video(video)
    cur_video.vid_warning.empty()

# Object Detection tab
detection.header("Object Detection:")
output_container = detection.container()
output_text = output_container.empty()
output_vid = output_container.empty()
output_warning = output_container.empty()

det_but_space = detection.empty()
det_but = det_but_space.button("detect", help='Start detection')

if det_but and video is not None:
    with st.spinner('Processing...'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())

        vf = cv2.VideoCapture(tfile.name)
    detection.balloons()



    # Download button
    detection.download_button(
        label="Download detected video as mp4",
        data=pred_img_BufferedReader,
        file_name='Detected_video.mp4',
        mime="video/mp4"
    )
    det_but_space.empty()

elif det_but:
    output_warning.error("No input video!")
