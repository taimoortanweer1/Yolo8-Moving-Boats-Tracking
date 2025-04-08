import cv2
import numpy as np

# Video parameters
width, height = 5000, 288
fps = 30
duration = 6000  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('3objects_white_black_5000.mp4', fourcc, fps, (width, height))

# Create gradient background
background = np.zeros((height, width, 3), dtype=np.uint8)
#for x in range(width):
#    intensity = int(50 + 50 * (x / width))
#    background[:, x] = (intensity, intensity, intensity)

# Define the red color (BGR format)
red_color = (255, 255, 255)

# Object properties [current_x, start_x, end_x, speed, y_pos, direction, width, height]
objects = [
    [100, 100, 4800, 1.0, height // 4, 1, 100, 50],   # Red rectangle (top)
    [4800, 4800, 100, 1.0, height // 2, 1, 50, 50],  # Red rectangle (middle)
    [500, 500, 4800, 2.0, 3 * height // 4, 1, 100, 100]  # Red rectangle (bottom)
]

for frame in range(total_frames):
    frame_img = background.copy()

    for obj_data in objects:
        current_x, start_x, end_x, speed, y_pos, direction, obj_w, obj_h = obj_data

        # Update position
        new_x = current_x + speed * direction

        # Reverse direction at boundaries (To and Fro motion)
        if (direction > 0 and new_x >= end_x) or (direction < 0 and new_x <= start_x):
            direction *= -1

        # Update object data
        obj_data[0] = new_x
        obj_data[5] = direction

        # Draw red rectangle (BGR format)
        cv2.rectangle(frame_img, (int(new_x), int(y_pos)), (int(new_x) + obj_w, int(y_pos) + obj_h), red_color, -1)

    # Write frame
    out.write(frame_img)

    # Progress indicator
    if frame % 100 == 0:
        positions = [f"{int(obj[0])}" for obj in objects]
        print(f"Frame {frame}/{total_frames} | Positions: {', '.join(positions)}")

out.release()
print("Animation saved as 'dual_object_animation.mp4'")
