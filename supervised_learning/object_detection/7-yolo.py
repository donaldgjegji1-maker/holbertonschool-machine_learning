#!/usr/bin/env python3
"""
YOLOv3 object detection module

This module provides the Yolo class for object detection
"""

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

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model

        Args:
            images: list of images as numpy.ndarrays

        Returns:
            tuple of (pimages, image_shapes)
        """
        # Get model input shape
        input_shape = self.model.input.shape[1:3]
        input_h, input_w = input_shape

        # Initialize arrays
        ni = len(images)
        pimages = np.zeros((ni, input_h, input_w, 3))
        image_shapes = np.zeros((ni, 2))

        # Process each image
        for i, image in enumerate(images):
            # Store original shape
            height, width = image.shape[0], image.shape[1]
            image_shapes[i] = [height, width]

            # Resize image using inter-cubic interpolation
            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values to [0, 1]
            resized = resized.astype(np.float32) / 255.0

            # Store preprocessed image
            pimages[i] = resized

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display image with bounding boxes, class names, and scores

        Args:
            image: numpy.ndarray containing unprocessed image
            boxes: numpy.ndarray containing boundary boxes
            box_classes: numpy.ndarray containing class indices
            box_scores: numpy.ndarray containing box scores
            file_name: file path where original image is stored
        """
        # Make a copy of the image to draw on
        img = image.copy()

        # Draw each box
        for i in range(len(boxes)):
            # Get box coordinates (convert to integers)
            x1, y1, x2, y2 = boxes[i].astype(int)

            # Draw blue rectangle (thickness 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Get class name and score
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]

            # Create text string
            text = f"{class_name} {score:.2f}"

            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Position text 5 pixels above top-left corner
            text_x = x1
            text_y = y1 - 5

            # If text would go above image, put it inside the box
            if text_y - text_height < 0:
                text_y = y1 + text_height + 5

            # Draw red text
            cv2.putText(img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # Display the image
        cv2.imshow(file_name, img)

        # Wait for key press
        key = cv2.waitKey(0)

        # Check if 's' key was pressed
        if key == ord('s'):
            # Create detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Save the image
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, img)

        # Close the window
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict objects in all images in a folder

        Args:
            folder_path: string representing path to folder with images

        Returns:
            tuple of (predictions, image_paths)
        """
        # Load all images
        images, image_paths = self.load_images(folder_path)

        # Preprocess images
        pimages, image_shapes = self.preprocess_images(images)

        # Make predictions
        outputs = self.model.predict(pimages)

        # Initialize predictions list
        predictions = []

        # Process each image
        for i in range(len(images)):
            # Get outputs for this image (each output is a batch with 1 image)
            image_outputs = [output[i:i+1] for output in outputs]

            # Process outputs
            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs, image_shapes[i])

            # Filter boxes
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)

            # Apply non-max suppression
            fin_boxes, final_classes, final_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)

            # Get filename without path for display
            file_name = os.path.basename(image_paths[i])

            # Display results
            self.show_boxes(images[i], fin_boxes, final_classes,
                            final_scores, file_name)

            # Store predictions
            predictions.append((fin_boxes, final_classes, final_scores))

        return predictions, image_paths
