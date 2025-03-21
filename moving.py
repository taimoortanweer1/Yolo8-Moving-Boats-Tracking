import cv2
import numpy as np
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracking algorithm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Load a pre-trained object detection model (e.g., Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Adjust max_age for tracking duration

# Open the grayscale video
video_path = r'c:\\Users\\CSDI\\Desktop\\stbrdfire_5_oct_24.avi'  # Update this with the correct video file path
video_path = r'c:\\Users\\CSDI\\Desktop\\1.mp4'  # Update this with the correct video file path

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Process each frame
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert grayscale frame to 3-channel image (required by the model)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Preprocess the frame for the model
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_tensor = transform(frame).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract detected objects (bounding boxes, labels, scores)
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter detections based on confidence threshold
    confidence_threshold = 0.5
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # Prepare detections for DeepSORT
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))  # Convert to [x, y, w, h] format

    # Update tracker with detected objects
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects with IDs on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()  # Get bounding box in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video

    # Display the frame (optional)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Release resources
cap.release()
cv2.destroyAllWindows()