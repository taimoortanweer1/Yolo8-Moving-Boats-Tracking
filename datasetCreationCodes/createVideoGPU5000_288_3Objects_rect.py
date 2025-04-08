import cv2
import numpy as np

# Video parameters
width, height = 3700, 288
fps = 30
duration = 60  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dual_object_animation_gpu.mp4', fourcc, fps, (width, height))

# Create gradient background
background = np.zeros((height, width, 3), dtype=np.uint8)
for x in range(width):
    intensity = int(50 + 50 * (x / width))
    background[:, x] = (intensity, intensity, intensity)

# Convert background to GPU
background_gpu = cv2.cuda_GpuMat()
background_gpu.upload(background)

# Define the red color (BGR format)
red_color = (0, 0, 255)

# Object properties [current_x, start_x, end_x, speed, y_pos, direction, width, height]
objects = [
    [100, 100, 3500, 3.0, height // 4, 1, 200, 50],   # Red rectangle (top)
    [3500, 3500, 100, 3.0, height // 2, -1, 200, 50],  # Red rectangle (middle)
    [500, 500, 3200, 2.0, 3 * height // 4, 1, 250, 60]  # Red rectangle (bottom)
]

# GPU processing for the rectangles (will use cuda in drawing)
for frame in range(total_frames):
    frame_gpu = background_gpu.clone()

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

        # Create a red rectangle on GPU
        rect_gpu = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(rect_gpu, (int(new_x), int(y_pos)), (int(new_x) + obj_w, int(y_pos) + obj_h), red_color, -1)
        
        # Upload to GPU
        rect_gpu_gpu = cv2.cuda_GpuMat()
        rect_gpu_gpu.upload(rect_gpu)

        # Blend the red rectangle onto the frame
        cv2.cuda.add(frame_gpu, rect_gpu_gpu, frame_gpu)

    # Download the frame from GPU
    frame_img = frame_gpu.download()

    # Write frame to video
    out.write(frame_img)

    # Progress indicator
    if frame % 100 == 0:
        positions = [f"{int(obj[0])}" for obj in objects]
        print(f"Frame {frame}/{total_frames} | Positions: {', '.join(positions)}")

out.release()
print("Animation saved as 'dual_object_animation_gpu.mp4'")
