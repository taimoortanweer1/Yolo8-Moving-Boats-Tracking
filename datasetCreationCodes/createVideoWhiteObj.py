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

# Rectangle parameters
rect_w, rect_h = 50, 30  # Width and height of the white rectangle

# Initial position (centered)
x, y = (width - rect_w) // 2, (height - rect_h) // 2

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
    if x <= 0 or x >= width - rect_w:
        direction_x *= -1
    if y <= 0 or y >= height - rect_h:
        direction_y *= -1
    
    # Draw white rectangle
    frame_img[int(y):int(y)+rect_h, int(x):int(x)+rect_w] = (255, 255, 255)
    
    # Write frame
    out.write(frame_img)
    
    # Optional: Display progress
    if frame % 100 == 0:
        print(f"Rendering frame {frame}/{total_frames}")

out.release()
print("Video saved as 'moving_object.mp4'")