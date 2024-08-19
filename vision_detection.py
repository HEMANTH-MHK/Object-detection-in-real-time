from vidgear.gears import CamGear
import numpy as np
import cv2

# Load YOLOv4 model
yolov4_cfg_path = '/head_count_model/yolov4.cfg'
yolov4_weights_path = '/head_count_model/yolov4.weights'
yolov4_names_path = '/head_count_model/coco.names'

# Verify paths
import os
assert os.path.exists(yolov4_cfg_path), f"Config file not found: {yolov4_cfg_path}"
assert os.path.exists(yolov4_weights_path), f"Weights file not found: {yolov4_weights_path}"
assert os.path.exists(yolov4_names_path), f"Names file not found: {yolov4_names_path}"

# Load YOLOv4 network
net = cv2.dnn.readNetFromDarknet(yolov4_cfg_path, yolov4_weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(yolov4_names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize video stream
stream = CamGear(source='https://www.youtube.com/live/ZxL5Hm3mIBk?si=zdcsBkBxd6P-fMd3?si=qIi6M3tm8aUs92Fl', stream_mode=True, logging=True).start()

count = 0
while True:
    frame = stream.read()
    if frame is None:
        break

    count += 1
    if count % 6 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Extract bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []
    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Count cars and persons
    car_count = sum(1 for i in class_ids if classes[i] == 'car')
    person_count = sum(1 for i in class_ids if classes[i] == 'person')
    cv2.putText(frame, f"Cars: {car_count}", (50, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(frame, f"Persons: {person_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()

