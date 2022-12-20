import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


class YOLO_Model:
    def __init__(self):
        with open('data.yaml', mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # TODO: Change the directory name accordingly
        self.model = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')

