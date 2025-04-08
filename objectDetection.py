from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_objects_on_black(image_path, model_name='yolov8x.pt', threshold=30):
    """
    Detect bright objects on black background using YOLO with preprocessing.
    
    Args:
        image_path (str): Path to the input image
        model_name (str): YOLO model name (default: yolov8n.pt)
        threshold (int): Brightness threshold for black background detection (0-255)
    """
    # Load the YOLO model
    model = YOLO(model_name)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale for background analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for black background areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Apply mask to highlight objects on black background
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Enhance contrast for better detection
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Perform object detection on preprocessed image
    results = model(enhanced)
    
    # Filter results to only include detections on black background
    filtered_results = []
    for r in results:
        new_boxes = []
        for box in r.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Check if the detection is on black background
            roi = gray[y1:y2, x1:x2]
            if np.mean(roi) > threshold:  # Object is brighter than background
                new_boxes.append(box)
        
        # Update results with filtered boxes
        if new_boxes:
            r.boxes = new_boxes
            filtered_results.append(r)
    
    # Display original and processed images
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(masked_image)
    plt.title('Masked Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    if filtered_results:
        im_array = filtered_results[0].plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.imshow(im_rgb)
        plt.title('Detected Objects')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('No Objects Detected')
    plt.axis('off')
    
    plt.show()
    
    # Print detection results
    if filtered_results:
        print("Detected objects on black background:")
        for box in filtered_results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            print(f"- {class_name}: {confidence:.2f}")
    else:
        print("No objects detected on black background.")

if __name__ == "__main__":
    # Example usage
    input_image = "dataset\\detect_3_obj.png"  # Replace with your image path
    detect_objects_on_black(input_image, threshold=40)  # Adjust threshold as needed