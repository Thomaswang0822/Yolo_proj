from prediction import Yolo_Predictor

if __name__ == "__main__":
    YOLO_DIR = "."
    MODEL_DIR = "./best.onnx"
    fname = "background.jpg"
    # fname = "000229.jpg"

    yolo_model = Yolo_Predictor(MODEL_DIR, YOLO_DIR)
    img_input = yolo_model.load_img(fname)
    pred = yolo_model.one_prediction(img_input)
    yolo_model.NMS_Draw(pred, img_input)
    yolo_model.display_img()