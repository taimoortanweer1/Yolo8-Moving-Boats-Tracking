import cv2
import torch
import numpy as np

# Initialize the YOLOv5 model from the Ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load pre-trained YOLOv5 model

# Function to calculate IOU between two bounding boxes
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2
    
    # Calculate area of overlap
    x_left = max(x1, x1_b)
    y_top = max(y1, y1_b)
    x_right = min(x2, x2_b)
    y_bottom = min(y2, y2_b)
    
    # No overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Area of each box
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_b - x1_b) * (y2_b - y1_b)
    
    # IOU calculation
    return overlap_area / float(area_box1 + area_box2 - overlap_area)

# Function to track objects across frames
def track_objects(detections, tracked_objects):
    new_tracked_objects = []
    for det in detections:
        matched = False
        for track in tracked_objects:
            if iou(det[:4], track[:4]) > 0.5:  # IOU threshold of 0.5
                track[4] = det[4]  # Update object position
                new_tracked_objects.append(track)
                matched = True
                break
        if not matched:
            # If no match, add as new object with a unique ID
            tracked_objects.append([det[0], det[1], det[2], det[3], len(tracked_objects)])
            new_tracked_objects.append(tracked_objects[-1])
    return new_tracked_objects

# Open the video file (grayscale video)
video_path = r'c:\\Users\\CSDI\\Desktop\\3.mp4'  # Update this with the correct video file path
video_path = r'c:\\Users\\CSDI\\Desktop\\stbrdfire_5_oct_24.avi'  # Update this with the correct video file path

cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Initialize an empty list for tracked objects
tracked_objects = []

# Loop through each frame
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame back to BGR for YOLOv5
    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Run YOLOv5 for object detection
    results = model(color_frame)  # Results format: [x1, y1, x2, y2, confidence, class]

    # Get detections from YOLOv5
    detections = results.xyxy[0].cpu().numpy()
    
    # Prepare detections with format [x1, y1, x2, y2, confidence]
    new_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, _ = det
        new_detections.append([x1, y1, x2, y2, conf])

    # Track objects by comparing current frame detections with previous ones
    tracked_objects = track_objects(new_detections, tracked_objects)

    # Draw bounding boxes for tracked objects
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color box
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with tracked objects
    cv2.imshow('Object Tracking', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
