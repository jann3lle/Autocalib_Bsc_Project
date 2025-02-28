import os
import random
import numpy as np
import cv2 as cv
from pathlib import Path

# All helper functions remain unchanged except for fixes
def resize_image(img, width=800):
    height = int((width / img.shape[1]) * img.shape[0])
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def blur_image(img_resized):
    blur = cv.GaussianBlur(img_resized, (5, 5), 0)
    ret, thresh = cv.threshold(blur, 35, 255, cv.THRESH_BINARY)
    return thresh

def find_connected_components(thresh, connectivity=8):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity)
    return num_labels, labels, stats, centroids

def filter_components(centroids, thresh, stats, labels, num_labels, min_width=400, min_height=400, min_area=3000):
    mask = np.zeros_like(labels, dtype="uint8")
    color_copy = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        if w > min_width and h > min_height and area > min_area:
            component_mask = (labels == i).astype("uint8") * 255
            mask = cv.bitwise_or(mask, component_mask)
            cv.rectangle(color_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.circle(color_copy, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    return mask, color_copy

def clean_mask(mask, kernel_size=(13, 13)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    return cv.morphologyEx(mask, cv.MORPH_ELLIPSE, kernel)

def detect_circles(edges, min_radius=200, max_radius=600, adjusted_dp=1.4, adjusted_param2=10):
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=adjusted_dp, minDist=600,
                               param2=adjusted_param2, minRadius=min_radius, maxRadius=max_radius)
    return circles

def draw_circles(img_resized, circles, radius_text):
    if circles is not None:
        circles = circles[0, :].astype("int")
        for (x, y, r) in circles:
            cv.circle(img_resized, (x, y), r, (0, 255, 0), 2)
            cv.circle(img_resized, (x, y), 2, (0, 0, 255), 3)
    cv.putText(img_resized, radius_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_resized

def process_and_create_video():
    save_dir = Path(__file__).resolve().parent.parent.joinpath('data', 'processed', 'video-files')
    save_dir.mkdir(parents=True, exist_ok=True)
    my_dir = Path(__file__).resolve().parent
    img_dir = my_dir.parent.joinpath('data', 'raw')
    output_video_path = save_dir.joinpath("radius_video.mp4")
    
    frame_width, frame_height, fps, frame_duration = 800, 600, 24, 2
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    
    zoom_levels = ['z1_liver', 'z1_white', 'z2_liver', 'z2_white', 'z3_liver', 'z3_white', 'z4_liver', 'z4_white', 'z5_liver', 'z5_white']
    expected_radii = {f'z{i}_liver': 207 + 51 * (i - 1) for i in range(1, 6)}
    expected_radii.update({f'z{i}_white': 207 + 51 * (i - 1) for i in range(1, 6)})
    
    selected_images = []
    for zoom_level in zoom_levels:
        img_paths = list(img_dir.glob(f'{zoom_level}/*.png'))
        if img_paths:
            selected_images.extend(random.sample(img_paths, min(len(img_paths), 1)))
    selected_images = random.sample(selected_images, min(len(selected_images), 5))
    
    for img_pth in selected_images:
        try:
            zoom_level = img_pth.parent.name
            expected_radius = expected_radii.get(zoom_level, 'N/A')
            img = cv.imread(str(img_pth), cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] Unable to load image {img_pth}")
                continue

            img_resized = resize_image(img)
            blurred = blur_image(img_resized)
            num_labels, labels, stats, centroids = find_connected_components(blurred)
            mask, color_copy = filter_components(centroids, blurred, stats, labels, num_labels)
            mask_cleaned = clean_mask(mask)
            edges = cv.Canny(mask_cleaned, 50, 150)
            
            circles = detect_circles(edges)
            detected_radius = circles[0, 0, 2] if circles is not None else 'N/A'
            radius_text = f"Zoom: {zoom_level}, Expected Radius: {expected_radius}px, Detected Radius: {detected_radius}px"
            color_output = draw_circles(img_resized, circles, radius_text)
            color_output = cv.resize(color_output, (frame_width, frame_height))
            for _ in range(frame_duration * fps):
                out.write(color_output)

        except Exception as e:
            print(f"[ERROR] Exception processing {img_pth}: {e}")

    out.release()
    print(f"Video saved as {output_video_path}")

    cap = cv.VideoCapture(str(output_video_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("Radius-Video", frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    process_and_create_video()
