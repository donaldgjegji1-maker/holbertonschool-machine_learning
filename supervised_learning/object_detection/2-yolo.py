#!/usr/bin/env python3
"""YOLO v3 object detection - filter_boxes"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to a Darknet Keras model
        classes_path: path to a list of class names
        class_t: box score threshold for initial filtering
        nms_t: IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                 containing all anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process outputs from the Darknet model for a single image.

        outputs: list of numpy.ndarrays, each of shape
                 (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
        image_size: numpy.ndarray [image_height, image_width]

        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        # Input dimensions of the model (assumed square)
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # --- Extract raw predictions ---
            t_xy = output[..., :2]    # (grid_h, grid_w, anchors, 2)
            t_wh = output[..., 2:4]   # (grid_h, grid_w, anchors, 2)
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            # --- Apply sigmoid to xy and confidence ---
            b_xy = 1 / (1 + np.exp(-t_xy))   # sigmoid
            b_conf = 1 / (1 + np.exp(-box_conf))
            b_class_probs = 1 / (1 + np.exp(-class_probs))

            # --- Build cx, cy grid offsets ---
            # cx: column index, cy: row index
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            # b_x = sigmoid(t_x) + cx  (normalized by grid)
            b_x = (b_xy[..., 0] + cx) / grid_width
            b_y = (b_xy[..., 1] + cy) / grid_height

            # --- Anchor dimensions (for this output scale) ---
            pw = self.anchors[i, :, 0]  # shape (anchor_boxes,)
            ph = self.anchors[i, :, 1]

            # b_w = pw * exp(t_w)  normalized by input size
            b_w = (pw * np.exp(t_wh[..., 0])) / input_width
            b_h = (ph * np.exp(t_wh[..., 1])) / input_height

            # --- Convert (bx, by, bw, bh) → (x1, y1, x2, y2) ---
            # Scale to original image dimensions
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            # Stack into (grid_h, grid_w, anchors, 4)
            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(b_conf)
            box_class_probs.append(b_class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes by score threshold.

        boxes: list of numpy.ndarrays of shape
               (grid_height, grid_width, anchor_boxes, 4)
        box_confidences: list of numpy.ndarrays of shape
                         (grid_height, grid_width, anchor_boxes, 1)
        box_class_probs: list of numpy.ndarrays of shape
                         (grid_height, grid_width, anchor_boxes, classes)

        Returns: (filtered_boxes, box_classes, box_scores)
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes:    numpy.ndarray of shape (?,)
            box_scores:     numpy.ndarray of shape (?,)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            # Box score = confidence * class probability
            # conf shape:  (grid_h, grid_w, anchors, 1)
            # probs shape: (grid_h, grid_w, anchors, classes)
            scores = conf * probs  # (grid_h, grid_w, anchors, classes)

            # Class with highest score per box
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)

            # Mask: keep only boxes whose best score exceeds threshold
            mask = best_score >= self.class_t

            # Apply mask — flatten surviving boxes
            filtered_boxes.append(box[mask])               # (N, 4)
            box_classes.append(best_class[mask])           # (N,)
            box_scores.append(best_score[mask])            # (N,)

        # Concatenate results from all output scales
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
