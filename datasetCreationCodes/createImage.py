import cv2
import numpy as np

# 1. Create black background (320x288)
background = np.zeros((288, 320, 3), dtype=np.uint8)  # Height, Width, Channels

# 2. Load three overlay images (replace paths with your files)
overlay_paths = ["dataset\\obj1.png", "dataset\\obj2.png", "dataset\\obj3.png"]  # Your image paths
positions = [(50, 30), (150, 80), (80, 150)]  # (x,y) positions for each object

for i, overlay_path in enumerate(overlay_paths):
    # Read image (with alpha channel if available)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    
    if overlay is None:
        print(f"Error: Could not load {overlay_path}")
        continue
    
    x, y = positions[i]
    
    # Handle transparency (4 channels) or normal (3 channels)
    if overlay.shape[2] == 4:  # With alpha channel
        # Extract RGB and alpha
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0  # Normalize to 0-1
        
        # Blend each channel
        for c in range(3):
            background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = (
                background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] * (1 - alpha) +
                overlay_rgb[:, :, c] * alpha
            )
    else:  # Without alpha channel
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay

# 3. Save and show result
cv2.imwrite("output_multi_objects.png", background)
print("Saved as 'output_multi_objects.png'")

cv2.imshow("Result", background)
cv2.waitKey(0)
cv2.destroyAllWindows()