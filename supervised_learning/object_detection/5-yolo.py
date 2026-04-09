#!/usr/bin/env python3
"""YOLOv3 object detection - preprocess_images"""

import tensorflow as tf
import numpy as np
import cv2
import os


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
        self.model = tf.keras.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

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

            box_xy = output[..., 0:2]
            box_wh = output[..., 2:4]
            box_confidence = output[..., 4:5]
            box_class_probs_i = output[..., 5:]

            anchors_i = self.anchors[i]

            col = np.arange(grid_width).reshape(1, grid_width, 1)
            row = np.arange(grid_height).reshape(grid_height, 1, 1)

            cx = (1 / (1 + np.exp(-box_xy[..., 0])) + col) / grid_width
            cy = (1 / (1 + np.exp(-box_xy[..., 1])) + row) / grid_height

            w = (anchors_i[..., 0] * np.exp(box_wh[..., 0])) / image_width
            h = (anchors_i[..., 1] * np.exp(box_wh[..., 1])) / image_height

            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height

            boxes_i = np.stack([x1, y1, x2, y2], axis=-1)
            box_confidences_i = 1 / (1 + np.exp(-box_confidence))
            box_class_probs_i = 1 / (1 + np.exp(-box_class_probs_i))

            boxes.append(boxes_i)
            box_confidences.append(box_confidences_i)
            box_class_probs.append(box_class_probs_i)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on box scores and threshold

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        all_boxes = []
        all_classes = []
        all_scores = []

        for i in range(len(boxes)):
            classes = box_class_probs[i].shape[-1]

            boxes_i = boxes[i].reshape(-1, 4)
            box_confidences_i = box_confidences[i].reshape(-1)
            box_class_probs_i = box_class_probs[i].reshape(-1, classes)

            box_scores_i = box_confidences_i.reshape(-1, 1) * box_class_probs_i
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_scores_i = np.max(box_scores_i, axis=-1)

            filter_mask = box_scores_i >= self.class_t

            all_boxes.append(boxes_i[filter_mask])
            all_classes.append(box_classes_i[filter_mask])
            all_scores.append(box_scores_i[filter_mask])

        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression to filter overlapping boxes

        Returns:
            tuple of (box_predictions, predicted_box_classes, box_scores)
        """
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in unique_classes:
            class_indices = np.where(box_classes == c)[0]
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            sorted_indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            # Use stable index tracking to correctly accumulate kept boxes
            indices = np.arange(len(class_boxes))
            kept_indices = []

            while len(indices) > 0:
                best_idx = indices[0]
                kept_indices.append(best_idx)

                if len(indices) == 1:
                    break

                best_box = class_boxes[best_idx]
                rest_idx = indices[1:]
                rest_boxes = class_boxes[rest_idx]

                ix1 = np.maximum(best_box[0], rest_boxes[:, 0])
                iy1 = np.maximum(best_box[1], rest_boxes[:, 1])
                ix2 = np.minimum(best_box[2], rest_boxes[:, 2])
                iy2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, ix2 - ix1)
                inter_h = np.maximum(0, iy2 - iy1)
                intersection = inter_w * inter_h

                best_area = ((best_box[2] - best_box[0]) *
                             (best_box[3] - best_box[1]))
                rest_area = ((rest_boxes[:, 2] - rest_boxes[:, 0]) *
                             (rest_boxes[:, 3] - rest_boxes[:, 1]))
                union = best_area + rest_area - intersection

                iou = intersection / union
                keep_mask = iou < self.nms_t
                indices = rest_idx[keep_mask]

            box_predictions.append(class_boxes[kept_indices])
            predicted_box_classes.append(
                np.full(len(kept_indices), c, dtype=box_classes.dtype))
            predicted_box_scores.append(class_scores[kept_indices])

        if box_predictions:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0)
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0)
        else:
            box_predictions = np.array([]).reshape(0, 4)
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load all images from a folder

        Args:
            folder_path: string representing the path to the folder

        Returns:
            tuple of (images, image_paths)
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp',
                                          '.tiff', '.tif')):
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model.

        Args:
            images: list of images as numpy.ndarrays

        Returns:
            tuple of (pimages, image_shapes):
                pimages:      numpy.ndarray of shape (ni, input_h, input_w, 3)
                              resized to model input dims and scaled to [0, 1]
                image_shapes: numpy.ndarray of shape (ni, 2) containing the
                              original [height, width] of each image
        """
        # Retrieve the model's required input dimensions
        # model.input.shape is (batch, height, width, channels)
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            # Record original dimensions (height, width)
            image_shapes.append(image.shape[:2])

            # Convert BGR (OpenCV default) → RGB before resizing
            # using numpy slicing to reverse the channel axis
            image_rgb = image[:, :, ::-1]

            # Resize with inter-cubic interpolation to (input_w, input_h)
            # cv2.resize takes (width, height) as dsize
            resized = cv2.resize(image_rgb, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            rescaled = resized / 255.0

            pimages.append(rescaled)

        # Stack into a single array of shape (ni, input_h, input_w, 3)
        pimages = np.array(pimages)

        # Shape (ni, 2)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
