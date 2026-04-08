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
            box_xy = output[..., 0:2]
            box_wh = output[..., 2:4]
            box_confidence = output[..., 4:5]
            box_class_probs_i = output[..., 5:]

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on box scores and threshold

        Args:
            boxes: list of numpy.ndarrays containing boundary boxes
            box_confidences: list of numpy.ndarrays containing confidences
            box_class_probs: list of numpy.ndarrays containing class probs

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        # Initialize lists to store all boxes, classes, and scores
        all_boxes = []
        all_classes = []
        all_scores = []

        # Process each output
        for i in range(len(boxes)):
            # Get the shapes
            classes = box_class_probs[i].shape[-1]

            # Reshape arrays to 2D for easier processing
            boxes_i = boxes[i].reshape(-1, 4)
            box_confidences_i = box_confidences[i].reshape(-1)
            box_class_probs_i = box_class_probs[i].reshape(-1, classes)

            # Calculate box scores
            box_scores_i = box_confidences_i.reshape(-1, 1) * box_class_probs_i

            # Find the class with maximum score for each box
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_scores_i = np.max(box_scores_i, axis=-1)

            # Apply threshold filter
            filter_mask = box_scores_i >= self.class_t

            # Filter boxes, classes, and scores
            filtered_boxes_i = boxes_i[filter_mask]
            filtered_classes_i = box_classes_i[filter_mask]
            filtered_scores_i = box_scores_i[filter_mask]

            # Append to overall lists
            all_boxes.append(filtered_boxes_i)
            all_classes.append(filtered_classes_i)
            all_scores.append(filtered_scores_i)

        # Concatenate all results
        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression to filter overlapping boxes

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4) containing boxes
            box_classes: numpy.ndarray of shape (?,) containing class numbers
            box_scores: numpy.ndarray of shape (?) containing box scores

        Returns:
            tuple of (box_predictions, predicted_box_classes, box_scores)
        """
        # Get unique classes
        unique_classes = np.unique(box_classes)

        # Initialize lists for results
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Process each class separately
        for c in unique_classes:
            # Get indices for this class
            class_indices = np.where(box_classes == c)[0]

            # Get boxes, scores for this class
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            # Sort by score in descending order
            sorted_indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            # Initialize list to keep boxes after NMS
            keep_indices = []

            # Apply NMS
            while len(class_boxes) > 0:
                # Keep the box with highest score
                keep_indices.append(0)

                if len(class_boxes) == 1:
                    break

                # Calculate IOU with remaining boxes
                box1 = class_boxes[0]
                rest_boxes = class_boxes[1:]

                # Calculate intersection coordinates
                x1 = np.maximum(box1[0], rest_boxes[:, 0])
                y1 = np.maximum(box1[1], rest_boxes[:, 1])
                x2 = np.minimum(box1[2], rest_boxes[:, 2])
                y2 = np.minimum(box1[3], rest_boxes[:, 3])

                # Calculate intersection area
                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                # Calculate box areas
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_rest = ((rest_boxes[:, 2] - rest_boxes[:, 0]) *
                             (rest_boxes[:, 3] - rest_boxes[:, 1]))

                # Calculate IOU
                iou = intersection / (area1 + area_rest - intersection)

                # Keep boxes with IOU less than threshold
                keep_mask = iou < self.nms_t
                class_boxes = rest_boxes[keep_mask]
                class_scores = class_scores[1:][keep_mask]

            # Add kept boxes to results
            if keep_indices:
                box_predictions.append(class_boxes[keep_indices])
                predicted_box_classes.append(np.full(len(keep_indices), c))
                predicted_box_scores.append(class_scores[keep_indices])

        # Concatenate results
        if box_predictions:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0)
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0)
        else:
            # Return empty arrays if no boxes
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

        # Get all files in the folder
        for filename in os.listdir(folder_path):
            # Check if file is an image (common extensions)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp',
                                          '.tiff', '.tif')):
                file_path = os.path.join(folder_path, filename)
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths
