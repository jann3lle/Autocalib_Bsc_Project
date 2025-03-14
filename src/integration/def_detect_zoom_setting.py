import numpy as np
import cv2 as cv
import os
import glob
from pathlib import Path

def detect_zoom_setting(img_path):
    ''' Final code with image processing steps to detect zoom level'''
    
    # Define expected radii
    base_radii = [207, 258, 309, 361, 410]

    def resize_image(img_path, width=800):
        ''' Function to resize the image'''
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        height = int((width / img.shape[1] * img.shape[0]))
        img_resized = cv.resize(img, (width, height))
        return img_resized


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
            #print(f"Component {i}: Bounding Box ({x}, {y}, {w}, {h}), Area: {area}, Centroid: ({cX:.2f}, {cY:.2f})")
            
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

    def detect_circles(edges, min_radius=200, max_radius=600, adjusted_dp = 1.4, adjusted_param2= 6):
        circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=adjusted_dp, minDist=600,
                                    param2=adjusted_param2, minRadius=min_radius, maxRadius=max_radius)
        num_circles =  circles.shape[1] # Number of circles detected
        
    
        if circles is not None:
            print(f"Number of circles detected: {num_circles}")

            circles = np.round(circles[0, :]).astype("int")  # Convert to integers
            for (x, y, r) in circles:
                print(f"x-coordinate: {x}, y-coordinate: {y}")

        else:
            print("No circles detected.")
        return circles
    
    def closest_zoom(circles, base_radii, tolerance = 0.1):
        ''' Determines the closest zoom level based on detected circle radius.
    
        Parameters:
        - circles: List of detected circles [(x, y, radius)].
        - base_radii: List of expected radii corresponding to zoom levels [1-5].
        - tolerance: Acceptable percentage deviation.
        
        Returns:
        - zoom_level: The closest matching zoom level (1 to 5).
        '''
        # Since only one circle is detected, get its radius
        detected_radius = circles[0][2]  
        print(f"Detected radius: {detected_radius}")

        # Find closest zoom level
        zoom_level = min(range(1, 6), key=lambda i: abs(detected_radius - base_radii[i - 1]))
        
        # Deviation from detected to actual radius
        expected_radius = base_radii[zoom_level - 1]
        print(f"Expected radius: {expected_radius}")
        deviation = abs(detected_radius - expected_radius)

        # Check if deviation is within a good range
        if deviation <= expected_radius * tolerance:
            print(f"Deviation: {deviation}px within tolerance")
        else:
            print(f"Deviation too large: {deviation}px out of tolerance")
        print(f"Detected Zoom Level: {zoom_level}")
        
        return zoom_level
    img_resized = resize_image(img)
    thresh = blur_image(img_resized)
    num_labels, labels, stats, centroids = find_connected_components(thresh)
    mask, color_copy = filter_components(centroids, thresh, stats, labels, num_labels)
    mask_cleaned = clean_mask(mask)
    edges = cv.Canny(mask_cleaned, 50, 150)
    circles = detect_circles(edges)
    zoom_level = closest_zoom(circles, base_radii)
    
    return zoom_level

def get_single_img(img_dir, zoom_num=1, type= "liver", img_path_no = 0):
    ''' Outputs a single image from any zoom level'''
    zoom_dir = img_dir.joinpath(f'z{zoom_num}_{type}')
    img_paths = list(zoom_dir.glob("*.png"))
    img_path = cv.imread(str(img_paths[img_path_no]))
    return img_path

if __name__ == "__main__":
    # Define image directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    img_dir = base_dir/ 'data' / 'raw' 
    img = get_single_img(img_dir)
    detect_zoom_setting(img)