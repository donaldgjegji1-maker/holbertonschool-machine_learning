#!/usr/bin/env python3
"""YOLO v3 object detection - non_max_suppression"""

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

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            b_xy = 1 / (1 + np.exp(-t_xy))
            b_conf = 1 / (1 + np.exp(-box_conf))
            b_class_probs = 1 / (1 + np.exp(-class_probs))

            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            b_x = (b_xy[..., 0] + cx) / grid_width
            b_y = (b_xy[..., 1] + cy) / grid_height

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            b_w = (pw * np.exp(t_wh[..., 0])) / input_width
            b_h = (ph * np.exp(t_wh[..., 1])) / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(b_conf)
            box_class_probs.append(b_class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes by score threshold.

        Returns: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, probs in zip(boxes, box_confidences, box_class_probs):
            scores = conf * probs
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)
            mask = best_score >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Max Suppression to filtered bounding boxes.

        filtered_boxes: numpy.ndarray of shape (?, 4)
        box_classes:    numpy.ndarray of shape (?,)
        box_scores:     numpy.ndarray of shape (?,)

        Returns: (box_predictions, predicted_box_classes, predicted_box_scores)
            All ordered by class first, then by descending box score.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Process each unique class independently
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Gather all boxes belonging to this class
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]    # (N, 4)
            cls_scores = box_scores[cls_mask]        # (N,)

            # Sort by score descending
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            # Greedy NMS
            keep = []
            while len(cls_boxes) > 0:
                # Always keep the highest-scoring box
                keep.append(0)

                if len(cls_boxes) == 1:
                    break

                # Compute IoU of best box vs all remaining boxes
                best = cls_boxes[0]
                rest = cls_boxes[1:]

                # Intersection coordinates
                ix1 = np.maximum(best[0], rest[:, 0])
                iy1 = np.maximum(best[1], rest[:, 1])
                ix2 = np.minimum(best[2], rest[:, 2])
                iy2 = np.minimum(best[3], rest[:, 3])

                # Intersection area (clamp to 0)
                inter_w = np.maximum(0, ix2 - ix1)
                inter_h = np.maximum(0, iy2 - iy1)
                intersection = inter_w * inter_h

                # Union area
                best_area = (best[2] - best[0]) * (best[3] - best[1])
                rest_area = (rest[:, 2] - rest[:, 0]) * \
                            (rest[:, 3] - rest[:, 1])
                union = best_area + rest_area - intersection

                iou = intersection / union

                # Keep only boxes with IoU below the threshold
                suppression_mask = iou < self.nms_t
                cls_boxes = cls_boxes[1:][suppression_mask]
                cls_scores = cls_scores[1:][suppression_mask]

            box_predictions.append(cls_boxes[:len(keep)])
            predicted_box_classes.append(
                np.full(len(keep), cls, dtype=box_classes.dtype))
            predicted_box_scores.append(cls_scores[:len(keep)])

        # Concatenate results across all classes
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores
