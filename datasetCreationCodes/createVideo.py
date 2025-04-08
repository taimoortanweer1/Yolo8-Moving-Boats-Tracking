import cv2
import numpy as np

# Video parameters
width, height = 320, 288
fps = 30
duration = 60  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('moving_object.mp4', fourcc, fps, (width, height))

# Load object (replace with your image path)
obj = cv2.imread('dataset\\obj1.png', cv2.IMREAD_UNCHANGED)
if obj is None:
    raise FileNotFoundError("Could not load object image")

# Initial position (centered)
obj_h, obj_w = obj.shape[:2]
x, y = (width - obj_w) // 2, (height - obj_h) // 2

# Movement parameters (adjust for speed/direction)
speed_x = 0.5  # pixels per frame
speed_y = 0.3
direction_x = 1  # 1 or -1
direction_y = 1

for frame in range(total_frames):
    # Create black background
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Update position (with boundary checking)
    x += speed_x * direction_x
    y += speed_y * direction_y
    
    # Reverse direction at edges
    if x <= 0 or x >= width - obj_w:
        direction_x *= -1
    if y <= 0 or y >= height - obj_h:
        direction_y *= -1
    
    # Handle transparency if exists
    if obj.shape[2] == 4:  # PNG with alpha
        alpha = obj[:, :, 3:] / 255.0
        obj_rgb = obj[:, :, :3]
        
        # Blend object onto background
        roi = frame_img[int(y):int(y)+obj_h, int(x):int(x)+obj_w]
        roi[:] = (1 - alpha) * roi + alpha * obj_rgb
    else:  # No alpha
        frame_img[int(y):int(y)+obj_h, int(x):int(x)+obj_w] = obj
    
    # Write frame
    out.write(frame_img)
    
    # Optional: Display progress
    if frame % 100 == 0:
        print(f"Rendering frame {frame}/{total_frames}")

out.release()
print("Video saved as 'moving_object.mp4'")