import cv2
import numpy as np

# Video parameters
width, height = 3700, 288
fps = 30
duration = 600  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dual_object_animation.mp4', fourcc, fps, (width, height))

# Create gradient background
background = np.zeros((height, width, 3), dtype=np.uint8)
#for x in range(width):
#    intensity = int(50 + 50 * (x / width))
#    background[:, x] = (intensity, intensity, intensity)

# Load objects (replace with your image paths)
obj1 = cv2.imread('dataset\\obj1.png', cv2.IMREAD_UNCHANGED)  # Right-moving object
obj2 = cv2.imread('dataset\\obj2.png', cv2.IMREAD_UNCHANGED)  # Left-moving object
obj3 = cv2.imread('dataset\\obj3.png', cv2.IMREAD_UNCHANGED)  # Example third object (you can add more)
if obj1 is None or obj2 is None or obj3 is None:
    raise FileNotFoundError("Could not load object images")

# Object properties [current_x, start_x, end_x, speed, y_pos, direction, object]
objects = [
    [100, 100, 3500, 3.0, height // 4, 1, obj1],    # Right-moving (top)
    [3500, 3500, 100, 3.0, height // 2, -1, obj2],  # Left-moving (middle)
    [500, 500, 3200, 2.0, 3 * height // 4, 1, obj3]  # Another object moving right (bottom)
]

for frame in range(total_frames):
    frame_img = background.copy()

    for obj_data in objects:
        current_x, start_x, end_x, speed, y_pos, direction, obj = obj_data
        obj_h, obj_w = obj.shape[:2]

        # Update position
        new_x = current_x + speed * direction

        # Reverse direction at boundaries (To and Fro motion)
        if (direction > 0 and new_x >= end_x) or (direction < 0 and new_x <= start_x):
            direction *= -1

        # Update object data
        obj_data[0] = new_x
        obj_data[5] = direction

        # Handle transparency
        if obj.shape[2] == 4:  # With alpha channel
            alpha = obj[:, :, 3:] / 255.0
            obj_rgb = obj[:, :, :3]

            # Calculate bounds
            x1, y1 = int(new_x), int(y_pos)
            x2, y2 = x1 + obj_w, y1 + obj_h

            # Clip to frame boundaries
            if x1 < 0:
                crop = -x1
                alpha = alpha[:, crop:]
                obj_rgb = obj_rgb[:, crop:]
                x1 = 0
            if x2 > width:
                crop = x2 - width
                alpha = alpha[:, :-crop]
                obj_rgb = obj_rgb[:, :-crop]
                x2 = width

            # Blend
            roi = frame_img[y1:y2, x1:x2]
            roi[:] = (1 - alpha) * roi + alpha * obj_rgb
        else:  # Without alpha
            frame_img[y_pos:y_pos + obj_h, int(new_x):int(new_x) + obj_w] = obj

    # Write frame
    out.write(frame_img)

    # Progress indicator
    if frame % 100 == 0:
        positions = [f"{int(obj[0])}" for obj in objects]
        print(f"Frame {frame}/{total_frames} | Positions: {', '.join(positions)}")

out.release()
print("Animation saved as 'dual_object_animation.mp4'")
