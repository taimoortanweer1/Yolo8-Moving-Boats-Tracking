import cv2
import numpy as np

# Video parameters
width, height = 3700, 288
fps = 30
duration = 120  # seconds
total_frames = fps * duration

# Create video writer (use 'XVID' for AVI or 'mp4v' for MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('wide_scroll_animation.mp4', fourcc, fps, (width, height))

# Load background (or create gradient)
background = np.zeros((height, width, 3), dtype=np.uint8)

# Create gradient background (optional)
for x in range(width):
    intensity = int(50 + 50 * (x/width))  # Gradient from dark to lighter gray
    background[:, x] = (intensity, intensity, intensity)

# Load moving object (replace with your image path)
obj = cv2.imread('dataset\\obj1.png', cv2.IMREAD_UNCHANGED)
if obj is None:
    raise FileNotFoundError("Could not load object image")

obj_h, obj_w = obj.shape[:2]
y_pos = (height - obj_h) // 2  # Center vertically

# Movement parameters
start_x, end_x = 100, 3500
speed = 3.0  # pixels per frame
current_x = start_x
direction = 1  # 1 = right, -1 = left

for frame in range(total_frames):
    # Create frame (copy background)
    frame_img = background.copy()
    
    # Update position
    current_x += speed * direction
    
    # Reverse direction at boundaries
    if current_x >= end_x or current_x <= start_x:
        direction *= -1
    
    # Handle transparency
    if obj.shape[2] == 4:  # PNG with alpha
        alpha = obj[:, :, 3:] / 255.0
        obj_rgb = obj[:, :, :3]
        
        # Calculate bounding box
        x1, y1 = int(current_x), y_pos
        x2, y2 = x1 + obj_w, y1 + obj_h
        
        # Ensure within frame bounds
        if x1 < 0:
            crop_left = -x1
            x1 = 0
            alpha = alpha[:, crop_left:]
            obj_rgb = obj_rgb[:, crop_left:]
        if x2 > width:
            crop_right = x2 - width
            x2 = width
            alpha = alpha[:, :-crop_right]
            obj_rgb = obj_rgb[:, :-crop_right]
        
        # Blend object
        roi = frame_img[y1:y2, x1:x2]
        roi[:] = (1 - alpha) * roi + alpha * obj_rgb
    else:  # No alpha
        frame_img[y_pos:y_pos+obj_h, int(current_x):int(current_x)+obj_w] = obj
    
    # Write frame
    out.write(frame_img)
    
    # Progress indicator
    if frame % 100 == 0:
        print(f"Rendering frame {frame}/{total_frames} - Position: {int(current_x)}")

out.release()
print("Animation saved as 'wide_scroll_animation.mp4'")