from ultralytics import YOLO
import cv2
# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")



# Open the video stream
cap = cv2.VideoCapture('c:\\Users\\CSDI\\Desktop\\stbrdfire_5_oct_24.avi')  # Use 0 for webcam or 'path_to_video.mp4' for a video file
cap = cv2.VideoCapture('c:\\Users\\CSDI\\Desktop\\3.mp4')  # Use 0 for webcam or 'path_to_video.mp4' for a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes
        classes = boxes.cls
        confidences = boxes.conf

        for box, cls, conf in zip(boxes.xyxy, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            confidence = float(conf)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label and confidence
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()