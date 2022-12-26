import streamlit as st

# TODO: Fill in app name
app_name = "My App"

# Set page configurations
# noinspection PyTypeChecker
st.set_page_config(
    page_title=app_name,
    page_icon="🉑",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# This is an *extremely* cool app!"
    }
)

# TODO: Header
st.header("Home Page")


# TODO: Project info
st.subheader("About our project:")
st.write("Our project reproduces [YOLO v5](https://github.com/ultralytics/yolov5) by training a new")

# TODO: Demo
st.subheader("Here is a demo of our object detection model:")


