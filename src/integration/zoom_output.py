import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import sys
import re

def load_images(img_dir):
    ''' Loads all images from the directory'''
    subfolders = sorted(img_dir.glob("z*"))

    all_zoom_levels = [] # Store all image paths

    # Iterate through subfolders and list images
    for subfolder in subfolders:
        img_paths = sorted(subfolder.glob("*.png"))
        #print(f"Folder: {subfolder.name}, Images: {[img.name for img in img_paths]}")
        all_zoom_levels.append(img_paths)

    if not all_zoom_levels:
        print("No images found in the directory")
    return all_zoom_levels

def choose_image(all_zoom_levels, chosen_img=None, chosen_zoom=None, chosen_category=None):
    ''' Selects an image based on the chosen frame number 
    all_zoom_levels = A list of all subfolder image lists
    zoom_level = A list of image paths for one subfolder
    img_path = A single image from zoom_level image path list
    '''
    # Choose frame_0{chosen_img}, within any of the subfolders
    for zoom_level in all_zoom_levels:
        for img_path in zoom_level:
            # Extract zoom level from file name
            match = re.search(r"z(\d+)", img_path.parent.stem)
            if match:
                detected_zoom = int(match.group(1))
            else:
                print("No zoom level found")
                continue

            # Extract category (e.g., "liver" or "white") from folder name
            category = img_path.parent.stem.split("_")[-1]  # Takes "liver" or "white" from "z2_liver"

            # Check if image matches the chosen frame AND optional zoom level    
            if f"frame_{chosen_img:03d}" in img_path.stem:
                if (chosen_zoom is None or detected_zoom == chosen_zoom) and\
                (chosen_category is None or category == chosen_category):
                    return img_path, detected_zoom, category
            
    raise ValueError(f"Image frame {chosen_img:03d} not found.")
    return None, None, None


DEBUG = None
def detect_zoom_setting(circles, detected_zoom, tolerance = 0.1, expected_radius = None):
    '''Detect and return zoom setting'''

    if circles is None:
        if DEBUG:
            print("[DEBUG] No circles detected")
        return False
    
    #circles = np.round(circles[0].astype("int"))
    if DEBUG:
        print(f"[DEBUG] Detected circles (x,y,r): {circles}")

    for (x,y,r) in circles:
        deviation = abs(r - expected_radius)
        if DEBUG:
            print(f"[DEBUG] Circle at ({x}, {y} with radius {r}. Expected radius {expected_radius})")

        if deviation <= expected_radius * tolerance:
            print(f"Expected radius: {expected_radius}")
            print(f"Deviation: {deviation}px")
        else:
            print("Deviation too large")
    return detected_zoom

def main():
    ''' main function'''
    # Load all image paths
    all_zoom_levels = load_images(img_dir)

    # Choose a specific image with category selection
    chosen_img = 10
    chosen_zoom = 5
    chosen_category = "liver"  # Example: Choose 'liver' category

    # Choose a specific image
    # Return value from choose_image function is equal to input_img_path
    
    input_img_path, detected_zoom, detected_category = choose_image(
        all_zoom_levels, chosen_img=chosen_img, chosen_zoom=chosen_zoom, chosen_category=chosen_category
    )

    # Print return value
    if input_img_path:
        print(f"Selected Image: {input_img_path} (Zoom: {detected_zoom})")
    
    # Define expected radii
    base_radii = [207, 258, 309, 361, 410]
    expected_radius = base_radii[detected_zoom - 1]
    
    from tests.circle_detection.circle_detection_func import resize_image, blur_image, find_connected_components,filter_components, clean_mask, detect_circles, draw_circles 
    img= cv2.imread(str(input_img_path), cv2.IMREAD_GRAYSCALE)
    # CALL THESE FUNCTIONS 
    img_resized = resize_image(img)
    thresh = blur_image(img_resized)
    num_labels, labels, stats, centroids = find_connected_components(thresh)
    mask, color_copy = filter_components(centroids, thresh, stats, labels, num_labels)
    mask_cleaned = clean_mask(mask)
    edges = cv2.Canny(mask_cleaned, 50, 150)
    circles = detect_circles(edges)
    draw_circles(img_resized, circles)
    
    # Call detect_zoom_setting function
    zoom_setting = detect_zoom_setting(circles, detected_zoom, expected_radius=expected_radius)
    print(f"Final detected zoom: {zoom_setting}")

if __name__ == "__main__":

    # Define image directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    img_dir = base_dir/ 'data' / 'raw' 
    sys.path.append(str(base_dir))

    main()