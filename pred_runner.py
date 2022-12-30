from prediction import Yolo_Predictor

if __name__ == "__main__":
    YOLO_DIR = "."
    MODEL_DIR = "./best.onnx"
    IMG_DIR = "./test_data"
    # Yolo will 'normalize' images of different sizes to 640x640. See <yolo dir>/export.py
    YOLO_IMG_WH=640 

    # fname = "background.jpg"
    fname = "000229.jpg"

    conf_thold=0.4  # how likely this bbox has bounded an object of interest
    prob_thold=0.5  # given this is an object of interest, what is the highest prob score of all classes

    yolo_model = Yolo_Predictor(MODEL_DIR, YOLO_DIR)
    img_input = yolo_model.load_data(fname, img_dir=IMG_DIR)
    pred = yolo_model.one_prediction(img_input)
    yolo_model.NMS_Draw(pred, img_input, conf_thold=conf_thold, prob_thold=prob_thold, YOLO_IMG_WH=YOLO_IMG_WH)
    yolo_model.display()