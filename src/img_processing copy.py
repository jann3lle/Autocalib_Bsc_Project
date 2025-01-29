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
    return output

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

def detect_circles(edges, min_radius=200, max_radius=600, adjusted_dp=1.4, adjusted_param2=10, zoom_level=None, expected_radii=None):
    # Detect circles using HoughCircles
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=adjusted_dp, minDist=600,
                               param2=adjusted_param2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        # Convert the circles to integer values
        circles = np.round(circles[0]).astype("int")
        print(f"Circles detected: {len(circles)}")
        
        # Display all detected circles' coordinates and radius
        for (x, y, r) in circles:
            print(f"Detected circle at ({x}, {y}) with radius {r}")
            if zoom_level in expected_radii:
                expected_radius = expected_radii[zoom_level]
                print(f"Expected radius: {expected_radius}px, Difference: {abs(expected_radius - r)}px")
        
        return circles  # Return all detected circles
    else:
        print("No circles detected.")
    return None

def draw_circles(img_resized, circles, zoom_level):
    if circles is not None:
        # Iterate over each circle and draw it
        for (x, y, r) in circles:
            # Draw the outer circle
            cv.circle(img_resized, (x, y), r, (255, 0, 255), 4)
            # Draw the center of the circle
            cv.circle(img_resized, (x, y), 2, (255, 0, 255), 3)
            # Add text showing the radius of the circle
            #cv.putText(img_resized, f'Radius: {r}px', (x - 40, y - r - 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(img_resized, f'Zoom Level: {zoom_level}',(x - 10, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print('No circles detected to draw.')
    return img_resized

def main():
    ''' Where __file__ rep the current file being worked on'''
    #print(__file__)
    my_dir = Path(__file__).resolve().parent # go up one level to tests folder, .resolve() alone gives the absolute location
    print(my_dir)
    img_dir = my_dir.parent.joinpath('data', 'raw', 'z1_white') #.parent - go up to main level (autocalib-for-mono)
    # then go into 'data/raw/z1_liver'
    print('Full path to zoom level subfolder:', img_dir)
    img_paths = img_dir.glob('*.png')  # Get all .png files in the directory
    # or img_paths = glob.glob(f'{img_dir}/*.png')  # Get all .png files as a list of strings
    #img_paths = list(img_dir.glob('*.png'))  # This will now be a list, not a map object

    # Specify which zoom level

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

    zoom_level = img_dir.name
    print(f"Processing images from zoom level: {zoom_level}")

    # Check if the zoom level exists in expected_radii
    if zoom_level in expected_radii:
        expected_radius = expected_radii[zoom_level]
        print(f"Expected radius for {zoom_level}: {expected_radius}px")
    else:
        print(f"Warning: {zoom_level} not found in expected radii dictionary.")

    for img_pth in img_paths:
    # Iterates through each file path in img_paths
        try:
            # Convert img_paths to a list
            img_paths = list(img_paths)

            if not img_paths:
                raise ValueError('No image files found in the directory.')
            
            # Use first image for processing
            first_img_pth = img_paths[0]

            # Read image
            img = cv.imread(first_img_pth, cv.IMREAD_GRAYSCALE)
        
            if img is None:
                raise ValueError (f'File {first_img_pth} could not be read')
            cv.imshow('test', img)
            cv.waitKey(1000)

        except Exception as e:
        # Handle any other unforeseen exceptions
            print(f"An unexpected error occurred while processing {img_pth}: {e}")

    cv.destroyAllWindows()
    
    # Save image 
    save_dir = Path(__file__).resolve().parent.parent.joinpath('data', 'processed', 'z5_debugging')  # Example output folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # Call 'Resize img' functrion
    img_resized = resize_image(img)
    cv.imshow('resized img',img_resized)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('original.png')
    cv.imwrite(str(output_image_path), img_resized)

    # Preprocess image (Gaussian blur and thresholding)
    thresh = blur_image(img_resized)
    cv.imshow('thresholded img',thresh)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('blurred.png')
    cv.imwrite(str(output_image_path), thresh)

    # Find connected components
    num_labels, labels, stats, centroids = find_connected_components(thresh, connectivity=8)

    # Filter components based on size criteria
    mask, color_copy = filter_components(centroids, thresh, stats, labels, num_labels, min_width=400, min_height=400, min_area=3000)
    cv.imshow('ctd comp img', color_copy)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('ctd-comp.png')
    cv.imwrite(str(output_image_path), color_copy)

    # Clean the mask using morphological operations
    mask_cleaned = clean_mask(mask)
    cv.imshow('clean mask', mask_cleaned)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('cleaned-mask.png')
    cv.imwrite(str(output_image_path), mask_cleaned)

    # Apply Canny edge detection
    edges = cv.Canny(mask_cleaned, 50, 150)
    cv.imshow('canny edge detected image', edges)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('canny-edge.png')
    cv.imwrite(str(output_image_path), edges)

    # Detect circles in the edge-detected image
    circles = detect_circles(edges, zoom_level=zoom_level, expected_radii=expected_radii)

    # Draw circles and display the result
    color_output = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)
    color_output = draw_circles(color_output, circles, zoom_level)
    cv.imshow('detected circle', color_output)
    cv.waitKey(1000)
    output_image_path = save_dir.joinpath('circle-detected.png')
    cv.imwrite(str(output_image_path), color_output)

if __name__ == '__main__':
    main()