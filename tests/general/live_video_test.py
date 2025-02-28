import os
import random
import cv2
from PIL import Image
from pathlib import Path
import numpy as np

def resize_image(img, width=800):
    ''' Function to resize the image '''
    height = int((width / img.shape[1]) * img.shape[0])
    return cv2.resize(img, (width, height))

# Save video
save_dir = Path(__file__).resolve().parent.parent.joinpath('data', 'processed', 'video-files')  # Example output folder
save_dir.mkdir(parents=True, exist_ok=True)

# Configuration
my_dir = Path(__file__).resolve().parent
image_folder = my_dir.parent.joinpath('data', 'raw')  # Replace with the path to your images
output_video_path = save_dir.joinpath("output_video_test.mp4")  # Name of the output video
frame_duration = 2  # Duration of each image in seconds
fps = 24  # Frames per second for the video

# Get list of image files, including from subfolders
image_files = []
for root, _, files in os.walk(image_folder):  # Recursively walk through directories
    for file in files:
        if file.lower().endswith('.png'):
            image_files.append(os.path.join(root, file))

# Check if there are enough images
if len(image_files) < 5:
    print("Not enough images in the folder!")
    exit()

# Select 5 random images
selected_images = random.sample(image_files, 5)

# Read and resize images
images = []
max_width = 800  # Example width for resizing

for file in selected_images:
    img = cv2.imread(file)
    resized_img = resize_image(img, width=max_width)
    images.append(resized_img)

# Get the resized dimensions for the video writer
standard_size = (images[0].shape[1], images[0].shape[0])

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(str(output_video_path), fourcc, fps, standard_size)

# Add each image to the video
for img in images:
    # Write the same frame multiple times to match frame duration
    for _ in range(frame_duration * fps):
        video.write(img)

# Release video writer
video.release()
print(f"Video created successfully: {output_video_path}")

# Play the video
cap = cv2.VideoCapture(str(output_video_path))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit the video early
        break

cap.release()
cv2.destroyAllWindows()
