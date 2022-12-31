import streamlit as st

# Set page configurations
# noinspection PyTypeChecker
st.set_page_config(
    page_title="Home",
    page_icon="😃",
    layout="centered",
    initial_sidebar_state="auto",
)

st.header("Home Page")

# TODO: Project info
st.subheader("About our project:")
st.write("[Our project](https://github.com/Thomaswang0822/Yolo_proj) is a reproduction of "
         "[YOLO v5](https://github.com/ultralytics/yolov5).")
st.write("**We can detect objects in the following classes:**")
st.write('car')
st.write('horse')
st.write('person')
st.write('bicycle')
st.write('cat')
st.write('dog')
st.write('train')
st.write('aeroplane')
st.write('dining table')
st.write('tv monitor')
st.write('chair')
st.write('bird')
st.write('bottle')
st.write('motorbike')
st.write('potted plant')
st.write('boat')
st.write('sofa')
st.write('sheep')
st.write('cow')
st.write('bus')

# TODO: Demo
st.subheader("Here is a demo of our object detection model:")

video_file = open('bar3.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes, format="video/mp4")