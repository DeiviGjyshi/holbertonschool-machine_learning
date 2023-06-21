#!/usr/bin/env python3
"""Object detection"""
import tensorflow.keras as K


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        with open(classes_path) as f:
            lines = f.readlines()
        classes = [x.strip() for x in lines]
        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
