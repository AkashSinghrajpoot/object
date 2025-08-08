import cv2
import numpy as np
# Load YOLO model
def load_yolo_model(cfg_file, weights_file, names_file):
    net = cv2.dnn.readNet(weights_file, cfg_file)  # Load the network
    with open(names_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]  # Read class labels
    return net, classes

# Function to detect objects using YOLO
def detect_objects(frame, net, classes):
    # Get the height and width of the frame
    height, width, _ = frame.shape
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Run the forward pass through YOLO network
    detections = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the threshold here
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                
                # Draw the bounding box and label
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return boxes, confidences, class_ids, indexes

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load YOLO model and classes
# Use raw string literals (prefix 'r') or escape the backslashes by using double backslashes

cfg_file = r'C:\Users\dell\Desktop\yolo4.cfg.txt'
weights_file = r'C:\Users\dell\Downloads\yolov4.weights'
names_file = r'C:\Users\dell\Desktop\coco.name.txt'

# Load YOLO model and classes
net, classes = load_yolo_model(cfg_file, weights_file, names_file)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Detect objects
    boxes, confidences, class_ids, indexes = detect_objects(frame, net, classes)

    # Draw bounding boxes for detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
