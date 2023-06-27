#!/usr/bin/env python3
"""Object detection"""
import glob
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        with open(classes_path, 'r') as f:
            classes_t = f.readlines()
        classes = [x.strip() for x in classes_t]
        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors        

    def sig(self, x):
        """sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """public method to process the output"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sig(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sig(output[..., 5:]))
        for i in range(len(boxes)):
            gridh = boxes[i].shape[0]
            gridw = boxes[i].shape[1]
            anchor = boxes[i].shape[2]
            t_x = boxes[i][..., 0]
            t_y = boxes[i][..., 1]
            t_w = boxes[i][..., 2]
            t_h = boxes[i][..., 3]
            box = np.zeros((gridh, gridw, anchor))
            indexX = np.arange(gridw).reshape(1, gridw, 1)
            indexY = np.arange(gridh).reshape(gridh, 1, 1)
            boxX = box + indexX
            boxY = box + indexY
            ntx = self.sig(t_x)
            nty = self.sig(t_y)
            bx = ntx + boxX
            by = nty + boxY
            bx = bx / gridw
            by = by / gridh
            anchorw = self.anchors[i, :, 0]
            anchorh = self.anchors[i, :, 1]
            bw = anchorw * np.exp(t_w)
            bh = anchorh * np.exp(t_h)
            inputw = self.model.input.shape[1].value
            inputh = self.model.input.shape[2].value
            bw = bw / inputw
            bh = bh / inputh
            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][..., 0] = x1 * image_size[1]
            boxes[i][..., 1] = y1 * image_size[0]
            boxes[i][..., 2] = x2 * image_size[1]
            boxes[i][..., 3] = y2 * image_size[0]
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes"""
        filtered_box = []
        box_scores = []
        box_classes = []
        for boxes_set, confidence_set, class_set in zip(boxes, box_confidences,
                                                        box_class_probs):
            scores = confidence_set * class_set
            classes = np.argmax(scores, axis=-1)
            max_score = np.max(scores, axis=-1)
            mask = max_score >= self.class_t
            filtered_box.extend(boxes_set[mask])
            box_classes.extend(classes[mask])
            box_scores.extend(max_score[mask])
        filtered_box = np.array(filtered_box)
        box_scores = np.array(box_scores)
        box_classes = np.array(box_classes)
        return filtered_box, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non max supression"""
        prdicted_boxes = []
        predicted_box_classes = []
        predicted_box_scores = []
        for c in set(box_classes):
            class_indices = np.where(box_classes == c, )[0]
            class_boxes = filtered_boxes[class_indices]
            class_socres = box_scores[class_indices]
            keep = self.nms(class_boxes, class_socres)
            prdicted_boxes.extend(class_boxes[keep])
            predicted_box_classes.extend(box_classes[class_indices][keep])
            predicted_box_scores.extend(class_socres[keep])
        prdicted_boxes = np.array(prdicted_boxes)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)
        return prdicted_boxes, predicted_box_classes, predicted_box_scores

    def nms(self, boxes, scores):
        """Perform non-max suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            indices = np.where(iou <= self.nms_t)[0]
            order = order[indices + 1]
        keep = np.array(keep)
        return keep 

    @staticmethod
    def load_images(folder_path):
        """Load images"""
        images_path = glob.glob(folder_path + '/*.jpg')
        images = [cv.imread(x) for x in images_path]
        return (images, images_path)

    def preprocess_images(self, images):
        """Preprocess images"""
        inputw = self.model.input.shape[1].value
        inputh = self.model.input.shape[2].value
        pimages = []
        image_shapes = []
        for i in range(len(images)):
            newX = images[i].shape[0]
            newY = images[i].shape[1]
            image_shapes.append((newX, newY))
            resize = cv.resize(images[i], (inputw, inputh),
                               interpolation=cv.INTER_CUBIC)
            resize = resize / 255
            pimages.append(resize)
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return (pimages, image_shapes)
