import cv2
import torch

# Load YOLOv5 model (small & fast)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_moving_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame (background should be black)
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference (black bg = no motion)
        diff = cv2.absdiff(prev_gray, gray)
        _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply YOLO only on moving regions (for efficiency)
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter small movements
                x, y, w, h = cv2.boundingRect(contour)
                
                # Crop moving object and run YOLO
                roi = frame[y:y+h, x:x+w]
                results = model(roi)
                
                # Draw YOLO detections on original frame
                frame[y:y+h, x:x+w] = results.render()[0]
        
        # Display output
        cv2.imshow("Moving Object Detection", frame)
        
        # Update previous frame
        prev_gray = gray.copy()
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run detection
if __name__ == "__main__":
    detect_moving_objects("dataset\\Objects_1_Back_Black_Time_01_W_320_1.mp4")  # Replace with your video path