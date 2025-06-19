import cv2
import serial
import numpy as np
from PIL import Image
import tensorflow as tf

# Load YOLOv3 model
model = tf.saved_model.load('yolov3.pb')

# Define serial port and baud rate
ser = serial.Serial('COM8', 115200, timeout=0)

# Define video capture object
cap = cv2.VideoCapture(ser.fileno())

# Define object detection parameters
conf_threshold = 0.5
iou_threshold = 0.4
class_names = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'aeroplane', 'train', 'boat', 'traffic light']

# Define function to perform object detection
def detect_objects(frame):
    # Preprocess frame
    input_frame = cv2.resize(frame, (416, 416))
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = np.expand_dims(input_frame, axis=3)

    # Perform object detection
    yolov3_outputs = model(np.array(input_frame))[0]
    boxes = yolov3_outputs['detection_boxes'][0].numpy()
    scores = yolov3_outputs['detection_scores'][0].numpy()
    classes = yolov3_outputs['detection_classes'][0].numpy()

    # Filter detections by confidence score
    filtered_boxes = []
    for i in range(len(boxes)):
        if scores[i] > conf_threshold:
            filtered_boxes.append([boxes[i], scores[i], classes[i]])

    # Filter detections by IOU threshold
    filtered_boxes = np.array(filtered_boxes)
    filtered_boxes = filter_boxes_iou(filtered_boxes, filtered_boxes)

    # Draw bounding boxes and labels
    for box, score, class_id in filtered_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1 * frame.shape[1], y1 * frame.shape[0], x2 * frame.shape[1], y2 * frame.shape[0]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_names[class_id]} {score:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_coordinates = (x1, y1 - label_size[1])
        cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, label_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Define function to filter bounding boxes by IOU threshold
def filter_boxes_iou(boxes, iou_threshold):
    filtered_boxes = []
    for i in range(len(boxes)):
        if i == 0:
            filtered_boxes.append(boxes[i])
            continue

        overlaps = []
        for j in range(len(filtered_boxes)):
            overlap = compute_iou(boxes[i], filtered_boxes[j])
            overlaps.append(overlap)

        if max(overlaps) < iou_threshold:
            filtered_boxes.append(boxes[i])