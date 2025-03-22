import cv2
import numpy as np
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracking algorithm
import os
import time

# Set environment variable to avoid library warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

model.eval()

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=3)  # Adjust max_age for tracking duration

# Video path
video_path = r'c:\\Users\\CSDI\\Desktop\\stbrdfire_5_oct_24.avi'  # Update this with the correct video file path

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Pre-processing transform once
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

frame_id = 0
start_time = time.time()  # Start time to calculate FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB if the frame is grayscale (ensure 3 channels)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Preprocess the frame
    img_tensor = transform(frame).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract detections
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter detections based on confidence threshold
    confidence_threshold = 0.5
    mask = scores > confidence_threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    # Prepare detections for DeepSORT
    detections = [
        ([x1, y1, x2 - x1, y2 - y1], score, label)
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels)
    ]

    # Update the tracker with detected objects
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            bbox = track.to_ltrb()  # Get bounding box in [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS (Optional)
    if frame_id % 30 == 0:  # Update FPS every 30 frames for smoother results
        elapsed_time = time.time() - start_time
        fps_est = frame_id / elapsed_time
        print(f"FPS: {fps_est:.2f}")

    # Display the frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
