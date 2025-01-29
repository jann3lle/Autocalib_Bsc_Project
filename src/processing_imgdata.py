# for loop used to process all images in each sub folder ( eg z1_liver)

import numpy as np
import cv2 as cv
import os
import glob
from pathlib import Path

def resize_image(img, width=800):
    ''' Function to resize the image'''
    height = int((width / img.shape[1] * img.shape[0]))
    return cv.resize(img, (width, height))
    # 2 arguments - input img, target size for img

def blur_image(img_resized):
    blur = cv.GaussianBlur(img_resized, (5, 5), 0)
    ret, thresh = cv.threshold(blur, 35, 255, cv.THRESH_BINARY)
    return thresh

def find_connected_components(thresh, connectivity=8):
    ''' This function finds connected components in the thresholded image. '''
    output = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
    num_labels, labels, stats, centroids = output
    #print(f'num_labels:{num_labels}, labels shape: {labels}, stats shape: {stats}')
    return num_labels, labels, stats, centroids

def filter_components(centroids,thresh, stats, labels, num_labels, min_width=400, min_height=400, min_area=3000):
    ''' Filter connected components based on size and area criteria. '''
    mask = np.zeros_like(labels, dtype="uint8")
    color_copy = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        #x, y, w, h, area = stats[i]
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        print(f"Component {i}: Bounding Box ({x}, {y}, {w}, {h}), Area: {area}, Centroid: ({cX:.2f}, {cY:.2f})")
        
        if w > min_width and h > min_height and area > min_area:
            component_mask = (labels == i).astype("uint8") * 255
            mask = cv.bitwise_or(mask, component_mask)
            
            # Draw bounding box and centroid on the image
            cv.rectangle(color_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Bounding box
            cv.circle(color_copy, (int(cX), int(cY)), 4, (0, 0, 255), -1)  # Centroid

    return mask, color_copy

def clean_mask(mask, kernel_size=(13, 13)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    return cv.morphologyEx(mask, cv.MORPH_ELLIPSE, kernel)

def detect_circles(edges, min_radius=200, max_radius=600, adjusted_dp = 1.4, adjusted_param2= 8):
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=adjusted_dp, minDist=600,
                                param2=adjusted_param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        print(f"Circles detected: {circles}")

    else:
        print("No circles detected.")
    return circles

def draw_circles(img_resized, circles):
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(img_resized, (x, y), r, (255, 0, 255), 4)
            cv.circle(img_resized, (x, y), 2, (255, 0, 255), 3)
            cv.putText(img_resized, f'Radius: {r}px', (x - 40, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print('No circle detected in draw_circles function')
    return img_resized

DEBUG = False # Set to True for detailed debugging

def calculate_success_rate(circles, expected_radius, tolerance=0.01):
    """
    Calculate success based on detected circle's radius and expected radius.
    :param circles: Detected circles from HoughCircles.
    :param expected_radius: The expected radius for the zoom level.
    :param tolerance: Allowed deviation percentage (default ±10%).
    :return: Boolean indicating success.
    """
    if circles is None:
        if DEBUG:
            print("[DEBUG] No circles detected")
        return False  # No circles detected
    
    circles = np.round(circles[0]).astype("int")  # Round circle values
    if DEBUG:
        print(f"[DEBUG] Detected circles (x,y,r): {circles}")

    matches = []
    for (x, y, r) in circles: # Uses r (radius) from detected circle as a comparative variable
        deviation = abs(r - expected_radius)
        if DEBUG:
            print(f"[DEBUG] Circle at ({x},{y} with radius {r}. Expected radius: {expected_radius}. Deviation: {deviation}")
        
        if deviation <= expected_radius * tolerance:
            matches.append(r)

    if DEBUG:
        print(f"[DEBUG] Total matches: {len(matches)} / {len(circles)}")

    return len(matches) > 0

def main():
    ''' Where __file__ rep the current file being worked on'''
    #print(__file__)
    my_dir = Path(__file__).resolve().parent # go up one level to tests folder, .resolve() alone gives the absolute location
    print(my_dir)
    img_dir = my_dir.parent.joinpath('data','raw') # form img_dir variable containing all images of all zoom levels
    print(img_dir)
    zoom_levels = ['z1_liver', 'z1_white', 'z2_liver', 'z2_white', 'z3_liver', 'z3_white', 'z4_liver', 'z4_white', 'z5_liver', 'z5_white']
    success_counts = {zoom: 0 for zoom in zoom_levels}
    total_counts = {zoom: 0 for zoom in zoom_levels}

    # Define expected radii:
    expected_radii = {
        'z1_liver': 207,
        'z1_white': 207,
        'z2_liver': 258,
        'z2_white': 258,
        'z3_liver': 309,
        'z3_white': 309,
        'z4_liver': 361,
        'z4_white': 361,
        'z5_liver': 410,
        'z5_white': 410,
    }

    # Process each zoom level
    for zoom_level in zoom_levels:
        print(f"Processing zoom level: {zoom_level}")
        img_paths = img_dir.glob(f'{zoom_level}/*.png') # Get all the .png files in the directory, and print out which zoom level we are working at

        for img_pth in img_paths:
            # Iterates through each file path in img_paths
            try:
                # Load and process each image
                img = cv.imread(str(img_pth), cv.IMREAD_GRAYSCALE)
                if img is None:
                    print(f" [ERROR] Unable to load image {img_pth}")
                    continue

                # Call image processing functions 
                img_resized = resize_image(img)
                thresh = blur_image(img_resized)
                num_labels, labels, stats, centroids = find_connected_components(thresh)
                mask, color_copy = filter_components(centroids, thresh, stats, labels, num_labels)
                mask_cleaned = clean_mask(mask)
                edges = cv.Canny(mask_cleaned, 50, 150)
                
                # Detect circles in the edge-detected image
                circles = detect_circles(edges)

                # Compare detected circles to expected radius
                expected_radius = expected_radii[zoom_level]
                success = calculate_success_rate(circles, expected_radius) # Calls function to calc success rate, with input variables circles and expected radius

                # Should calculate the success rate for each circle for each zoom level

                # Update success and total counts
                total_counts[zoom_level] += 1
                if success:
                    success_counts[zoom_level] += 1
                else:
                    print(f"[INFO] Failed to macth circles in {img_pth}")

                # Draw circles and display the result
                color_output = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)
                color_output = draw_circles(color_output, circles)
                #cv.imshow('detected circle', color_output)
                #cv.waitKey(10000)
        
            except Exception as e:
                print(f"[ERROR] Exception processing {img_pth}: {e}")

    # Calculate and display success rates
    print("\nSuccess Rates:")  
    for zoom_level in zoom_levels:
        total = total_counts[zoom_level]
        successes = success_counts[zoom_level]
        success_rate = (successes / total * 100) if total > 0 else 0
        print(f"{zoom_level}: {success_rate:.2f}%") 

if __name__ == '__main__':
    main()
