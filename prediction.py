import cv2
import yaml
from yaml.loader import SafeLoader
import os
from os.path import join as pjoin
import numpy as np 

class Yolo_Predictor():
    def __init__(self, onnx_dir, yolo_dir):
        # 1. Load data.yaml file (config); get id_tag map
        with open(pjoin(yolo_dir, 'data.yaml'), 'r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)     # dict
        self.labels = list(data_yaml['names'])
        # id-color map for drawing the bbox
        np.random.seed(777)
        self.colors = np.random.randint(0, 256, size=(len(self.labels), 3)).tolist()

        # 2. Load trained yolo model into opencv
        self.yolov5 = cv2.dnn.readNetFromONNX(onnx_dir)
        # set computing engine
        self.yolov5.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # make computations on device
        self.yolov5.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 3. Load testing images
    # *** We can have variable size training images, but test images should be cropped to fixed sqaure size ***
    def load_img(self, fname, img_dir="./test_data"):
        self.orig_image = cv2.imread(pjoin(img_dir, fname))  # original image (to be compared with)
        self.image = self.orig_image.copy()   # np ndarray
        nrow, ncol, ch = self.image.shape
        assert ch == 3, "require testing image in RGB format"

        l = max(nrow, ncol)
        img_input = np.zeros((l,l,3), dtype=np.uint8)   # black background board
        img_input[:nrow, :ncol] = self.image     # fill in image
        return img_input    # feed this to yolo model

    # for our web-app
    def load_img_from_stream(self, img_stream):
        self.orig_image = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)  # original image (to be compared with)
        self.orig_image = self.orig_image[:, :, ::-1]
        self.image = self.orig_image.copy()   # np ndarray
        nrow, ncol, ch = self.image.shape
        assert ch == 3, "require testing image in RGB format"

        l = max(nrow, ncol)
        img_input = np.zeros((l,l,3), dtype=np.uint8)   # black background board
        img_input[:nrow, :ncol] = self.image     # fill in image
        return img_input    # feed this to yolo model
    
    def one_prediction(self, img_input, YOLO_IMG_SZ = (640, 640) ):
        # YOLO_IMG_SZ: see export.py

        # See: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
        blob = cv2.dnn.blobFromImage(img_input, 1/255, YOLO_IMG_SZ, swapRB=True, crop=False)
        self.yolov5.setInput(blob)

        # shape = (1, #bbox, 25), 
        # 25 = 5 {centerX, centerY, w, h, confidence} + 20 {prob score of each class}
        pred = self.yolov5.forward()
        return pred[0] 

    # 5. Non-maximum Suppression: 
    ## filter predictions of a single image based on confidence and prob score threshold
    ## NMS removes duplicate bbox
    # 6. Draw bbox
    ## We want each bbox to have a tag (word instead of id) and a probability
    """ 5 && 6 """
    def NMS_Draw(self, pred, img_input, conf_thold=0.4, prob_thold=0.5, YOLO_IMG_WH=640):
        # factors to restore shape info of testing image, which is not 640x640
        x_factor, y_factor = img_input.shape[0]/YOLO_IMG_WH, img_input.shape[1]/YOLO_IMG_WH

        conf_list, bbox_list, id_list = [], [], []
        for row in pred:
            conf = row[4]   # confidence of "this bbox catch an object of whatever class"
            if conf > conf_thold:
                tag_id = row[5:].argmax()   # id (0-19) of "the most likely class/tag"
                prob_score = row[5:][tag_id]  # probability score of "the most likely class/tag"
                if prob_score >= prob_thold:
                    # bbox info
                    cx, cy, w, h = row[:4]
                    # denormalize (restore to int) top-left position, width, height
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    x_left = int( (cx - 0.5*w) * x_factor)
                    y_top = int( (cy - 0.5*h) * y_factor)
                    bbox = [x_left, y_top, width, height]

                    # store info
                    conf_list.append(conf)
                    bbox_list.append(bbox)
                    id_list.append(tag_id)

        # NMS
        nms_return = cv2.dnn.NMSBoxes(bbox_list, conf_list, conf_thold, prob_thold)
        if len(nms_return) == 0:
            print("No object of interest detected in the image.")
            return False
        index = nms_return.flatten()

        for idx in index:
            # retrieve info
            x_left, y_top, width, height = bbox_list[idx]
            conf = conf_list[idx] * 100
            tag = self.labels[ id_list[idx] ]

            # format text display
            text = f"{tag}: {conf:.1f}%"

            # cv2.rectangle(image, top-left, bot-right, box colr, thickness)
            GREEN = (0, 255, 0)
            BLACK = (0, 0, 0)
            cv2.rectangle(
                self.image, 
                (x_left, y_top), 
                (x_left+width, y_top+height), 
                self.colors[id_list[idx]], # color according to class id
                2
            )
            cv2.putText(self.image, text, (x_left, y_top-10), cv2.FONT_HERSHEY_PLAIN, 0.8, BLACK, 1)

        return True
    
    def display_img(self):
        cv2.imshow('Original Image', self.orig_image)
        cv2.imshow('With Object Detection', self.image)
        print("Press ESC to exit.")
        cv2.waitKey(27)
        cv2.destroyAllWindows()

