from prediction import Video_Predictor

if __name__ == "__main__":
    YOLO_DIR = "."
    MODEL_DIR = "./best.onnx"
    IMG_DIR = "./test_data"
    # Yolo will 'normalize' images of different sizes to 640x640. See <yolo dir>/export.py
    YOLO_IMG_WH=640 

    # fname = "background.jpg"
    fname = "train.mp4"

    conf_thold=0.4  # how likely this bbox has bounded an object of interest
    prob_thold=0.5  # given this is an object of interest, what is the highest prob score of all classes

    yolo_model = Video_Predictor(MODEL_DIR, YOLO_DIR)
    yolo_model.load_data(fname)
    yolo_model.video_detect(YOLO_IMG_SZ = (640, 640), 
            conf_thold=0.4, prob_thold=0.5)
    yolo_model.display()
    