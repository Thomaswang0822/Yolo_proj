import time
import streamlit as st
from PIL import Image
# Import prediction script 
from prediction import Yolo_Predictor

# Set page configurations
# noinspection PyTypeChecker
st.set_page_config(
    page_title="Detection",
    page_icon="🆒",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# This is an *extremely* cool app!"
    }
)

st.title("Object Detection")

cur_image = st.container()

# Current Input Image Displayer
cur_image.subheader("Your input image:")
cur_image.img_display = st.empty()
cur_image.img_warning = st.empty()
cur_image.img_warning.warning("No Image")

file_upload, camera, detection = st.tabs(["Upload", "Camera Input", "Object detection"])

# file_upload tab
file_upload.header("Upload Image")
file_upload.subheader("Please upload your image")
file_image = file_upload.file_uploader("One Image in .png or .jpg",
                                       type=['png', 'jpg'],
                                       accept_multiple_files=False,
                                       label_visibility="visible")
image = file_image
if file_image is not None:
    cur_image.img_display.image(image)
    cur_image.img_warning.empty()

# Camera input tab
camera.header("Camera Image Input")
camera_image = camera.camera_input("Take a picture with your camera:")


if camera_image is not None:
    image = camera_image
    cur_image.img_display.image(image)
    if file_image is not None:
        cur_image.img_warning.error("You have two image inputs! "
                                    "Only image from camera will be used if no further action is taken.")

# Object Detection tab
detection.header("Object Detection:")
output_container = detection.container()
output_text = output_container.empty()
output_img = output_container.empty()
output_warning = output_container.empty()

det_but = detection.button("detect",
                           help='Start detection')

if det_but and image is not None:
    with st.spinner('Processing...'):
        # TODO: Change this line to load the model and do the detection
        YOLO_DIR = "."
        MODEL_DIR = "./best.onnx"

        yolo_model = Yolo_Predictor(MODEL_DIR, YOLO_DIR)
        img_input = yolo_model.load_img_from_stream(image)
        pred = yolo_model.one_prediction(img_input)
        detected = yolo_model.NMS_Draw(pred, img_input)
        # yolo_model.display_img()

        # instead of display a pop-up window, we save the image. 
        pred_img = Image.fromarray(yolo_model.image, 'RGB')
        output_text.subheader("Image with object detection:")
        output_img.image(pred_img)
    detection.balloons()

elif det_but:
    output_warning.error("No input image!")
