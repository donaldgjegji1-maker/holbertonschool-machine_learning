#!/usr/bin/env python3
"""
YOLOv3 object detection module

This module provides the Yolo class for object detection
"""

import tensorflow as tf
import numpy as np


class Yolo:
    """YOLOv3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize YOLOv3 model for object detection

        Args:
            model_path: path to Darknet Keras model
            classes_path: path to list of class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Set thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model

        Args:
            outputs: list of numpy.ndarrays containing predictions
            image_size: numpy.ndarray containing [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract components
            box_xy = output[..., 0:2]  # (t_x, t_y)
            box_wh = output[..., 2:4]  # (t_w, t_h)
            box_confidence = output[..., 4:5]  # box confidence
            box_class_probs_i = output[..., 5:]  # class probabilities

            # Get anchors for this output
            anchors_i = self.anchors[i]

            # Create meshgrid for cell indices
            col = np.arange(grid_width).reshape(1, grid_width, 1)
            row = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Calculate bounding box coordinates
            # Center coordinates (cx, cy)
            cx = (1 / (1 + np.exp(-box_xy[..., 0])) + col) / grid_width
            cy = (1 / (1 + np.exp(-box_xy[..., 1])) + row) / grid_height

            # Width and height (w, h)
            w = (anchors_i[..., 0] * np.exp(box_wh[..., 0])) / image_width
            h = (anchors_i[..., 1] * np.exp(box_wh[..., 1])) / image_height

            # Convert to corner coordinates (x1, y1, x2, y2)
            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height
            # Stack boxes
            boxes_i = np.stack([x1, y1, x2, y2], axis=-1)

            # Apply sigmoid to box confidence
            box_confidences_i = 1 / (1 + np.exp(-box_confidence))

            # Apply sigmoid to class probabilities
            box_class_probs_i = 1 / (1 + np.exp(-box_class_probs_i))

            boxes.append(boxes_i)
            box_confidences.append(box_confidences_i)
            box_class_probs.append(box_class_probs_i)

        return boxes, box_confidences, box_class_probs
