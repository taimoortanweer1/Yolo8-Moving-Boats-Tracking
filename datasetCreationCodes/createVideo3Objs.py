import cv2
import numpy as np

# Video parameters
width, height = 320, 288
fps = 30
duration = 60  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('three_moving_objects.mp4', fourcc, fps, (width, height))

# Load three objects (replace with your image paths)
objects = [
    cv2.imread('dataset\\obj1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('dataset\\obj2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('dataset\\obj3.png', cv2.IMREAD_UNCHANGED)
]

# Verify all objects loaded
for i, obj in enumerate(objects):
    if obj is None:
        raise FileNotFoundError(f"Could not load object{i+1}.png")

# Object properties [x, y, speed_x, speed_y, direction_x, direction_y]
obj_props = [
    [width//4, height//4, 0.5, 0.3, 1, 1],      # Object 1
    [width//2, height//3, -0.4, 0.6, -1, 1],     # Object 2
    [3*width//4, 2*height//3, 0.3, -0.7, 1, -1]  # Object 3
]

for frame in range(total_frames):
    # Create black background
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, obj in enumerate(objects):
        # Get current properties
        x, y, sp_x, sp_y, dir_x, dir_y = obj_props[i]
        obj_h, obj_w = obj.shape[:2]
        
        # Update position
        new_x = x + sp_x * dir_x
        new_y = y + sp_y * dir_y
        
        # Boundary check and direction flip
        if new_x <= 0 or new_x >= width - obj_w:
            dir_x *= -1
        if new_y <= 0 or new_y >= height - obj_h:
            dir_y *= -1
            
        # Update properties
        obj_props[i] = [new_x, new_y, sp_x, sp_y, dir_x, dir_y]
        
        # Handle transparency
        if obj.shape[2] == 4:  # With alpha
            alpha = obj[:, :, 3:] / 255.0
            obj_rgb = obj[:, :, :3]
            roi = frame_img[int(new_y):int(new_y)+obj_h, int(new_x):int(new_x)+obj_w]
            roi[:] = (1 - alpha) * roi + alpha * obj_rgb
        else:  # Without alpha
            frame_img[int(new_y):int(new_y)+obj_h, int(new_x):int(new_x)+obj_w] = obj
    
    # Write frame
    out.write(frame_img)
    
    # Progress indicator
    if frame % 100 == 0:
        print(f"Rendering frame {frame}/{total_frames}")

out.release()
print("Video saved as 'three_moving_objects.mp4'")