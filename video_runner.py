from prediction import Video_Predictor
import cv2
import tempfile
import shutil
import os, subprocess

if __name__ == "__main__":
    YOLO_DIR = "."
    MODEL_DIR = "./best.onnx"
    IMG_DIR = "./test_data"
    # Yolo will 'normalize' images of different sizes to 640x640. See <yolo dir>/export.py
    YOLO_IMG_WH=640 

    fname = "train.mp4"

    conf_thold=0.4  # how likely this bbox has bounded an object of interest
    prob_thold=0.5  # given this is an object of interest, what is the highest prob score of all classes

    yolo_model = Video_Predictor(MODEL_DIR, YOLO_DIR)
    yolo_model.load_data(fname)
    valid = yolo_model.video_detect(YOLO_IMG_SZ = (640, 640), 
            conf_thold=0.4, prob_thold=0.5)
    print("FPS: ", yolo_model.fps)

    if not valid:
        print("No valid frame was caputured, thus no video to show.")
    else:
        # yolo_model.video_array contains all the frames (each w * h * 3 ndarray)
        # reconstruct a video from these frames and save to a video_out
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        fname = out_file.name
        writer = cv2.VideoWriter(
            out_file.name,
            cv2.VideoWriter_fourcc(*'mp4v'), 
            yolo_model.fps, 
            (yolo_model.w, yolo_model.h)
        )
        for frame in yolo_model.video_array:
            writer.write(frame)
        print("video output file size: ", os.stat(out_file.name).st_size )
        writer.release()

        out_file2 = tempfile.NamedTemporaryFile(suffix='.mp4')
        fname2 = out_file2.name
        subprocess.run(["ffmpeg", "-i", out_file.name,
                            "-r", "30", "-v", "16",
                            "-vcodec", "libx264", out_file2.name, '-y'])

        final_file = 'final_output.mp4'
        shutil.copy(fname2, final_file)

        # "-preset", "veryfast",

    