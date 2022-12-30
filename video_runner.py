from prediction import Video_Predictor
import cv2

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
    valid = yolo_model.video_detect(YOLO_IMG_SZ = (640, 640), 
            conf_thold=0.4, prob_thold=0.5)

    """         
    if valid:
        yolo_model.display()
    else:
        print("No video to show.") """

    if not valid:
        output_warning.warning("No valid frame was caputured, thus no video to show.")
    else:
        # yolo_model.video_array contains all the frames (each w * h * 3 ndarray)
        # reconstruct a video from these frames and save to a video_out
        # out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        writer = cv2.VideoWriter(
            "test_out.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), 
            yolo_model.fps, 
            (yolo_model.w, yolo_model.h)
        )
        for frame in yolo_model.video_array:
            writer.write(frame)
        writer.release()
    