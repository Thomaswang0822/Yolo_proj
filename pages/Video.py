import time
from io import BytesIO, BufferedReader

import cv2
import numpy as np
import streamlit as st
from PIL import Image
# Import prediction script
from prediction import Video_Predictor
import tempfile
import os, subprocess



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

# Current Input Video Displayer
cur_video.subheader("Your input video:")
cur_video.vid_display = st.empty()
cur_video.vid_warning = st.empty()
cur_video.vid_warning.warning("No Video")

file_upload, detection = st.tabs(["Upload Video", "Object detection"])

# file_upload tab
file_upload.header("Upload Video")
file_upload.subheader("Please upload your video")
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
    det_but_space.empty()
    with st.spinner('Processing...'):
        # grab the file
        tfile = tempfile.NamedTemporaryFile(suffix='.mp4')
        # save video stream to a tmp file
        tfile.write(video.read())

        YOLO_DIR = "."
        MODEL_DIR = "./best.onnx"

        yolo_model = Video_Predictor(MODEL_DIR, YOLO_DIR)
        yolo_model.load_data(tfile.name, v_dir='')
        print("Model has loaded video data.")
        valid = yolo_model.video_detect()
        print("Model has finished prediction.")
        if not valid:
            output_warning.warning("No valid frame was caputured, thus no video to show.")
        else:
            # yolo_model.video_array contains all the frames (each w * h * 3 ndarray)
            # reconstruct a video from these frames and save to a video_out
            
            # out_file = tempfile.NamedTemporaryFile(suffix='.mp4')
            out_file = "tmp1.mp4"
            writer = cv2.VideoWriter(
                out_file,
                cv2.VideoWriter_fourcc(*'mp4v'), 
                yolo_model.fps, 
                (yolo_model.w, yolo_model.h)
            )
            print("cv2 writer init DONE.")
            for frame in yolo_model.video_array:
                writer.write(frame)
            writer.release()
            print(f"Written to outfile1 {out_file}")
            print("outfile1 file size: ", os.stat(out_file).st_size )

            # Not all browsers support the codec
            # re-load the file and convert to a codec that is readable using ffmpeg 
            # out_file2 = tempfile.NamedTemporaryFile(suffix='.mp4')
            out_file2 = "tmp2.mp4"
            LOG_ERR = "16"
            os.system(f"ffmpeg -i {out_file} -r {yolo_model.fps} -vcodec libx264 {out_file2} -y")
            # subprocess.run(["ffmpeg", "-i", out_file,
            #                     "-r", str(yolo_model.fps), 
            #                      "-v", LOG_ERR,
            #                     "-vcodec", "libx264", out_file2, '-y'])

            # print(f"Written to outfile2 {out_file2}")
            # print("outfile2 file size: ", os.stat(out_file2).st_size )

            # display video_out on this web page
            output_text.subheader("Video with object detection:")
            with open(out_file, 'rb') as out2:
                output_vid.video(out2.read(), format="video/mp4")
    detection.balloons()




    """ Right click to download the video with prediction."""

elif det_but:
    output_warning.error("No input video!")
