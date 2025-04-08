
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # SORT tracking algorithm

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use other YOLOv8 models like yolov8s, yolov8m, etc.

# Initialize SORT tracker
tracker = Sort()

# Open the video file
video_path = 'c:\\Users\\CSDI\\Desktop\\stbrdfire_5_oct_24.avi'
video_path = 'c:\\Users\\CSDI\\Desktop\\1.mp4'

cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform object detection using YOLOv8
    results = model(gray_frame)

    # Extract detected objects
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])

    # Update SORT tracker with detections
    if len(detections) > 0:
        tracks = tracker.update(np.array(detections))
    else:
        tracks = tracker.update(np.empty((0, 5)))

    # Draw tracked objects on the grayscale frame
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(gray_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(gray_frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame (optional)
    cv2.imshow("Tracking", gray_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()