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
        circles = np.round(circles[0]).astype("int") #access first detected circle **need to improve
        print(f"Circles detected: {circles}")
        for (x,y,r) in circles:
            print(f"Detected circle radius: {r}")

    else:
        print("No circles detected.")
    return circles,r 

def draw_circles(img_resized, circles):
    if circles is not None:
        circles = np.round(circles[0]).astype("int")
        for (x, y, r) in circles:
            cv.circle(img_resized, (x, y), r, (255, 0, 255), 4)
            cv.circle(img_resized, (x, y), 2, (255, 0, 255), 3)
            cv.putText(img_resized, f'Radius: {r}px', (x - 40, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print('no circle detected in draw_circles')
    return img_resized

def main():
    ''' Where __file__ rep the current file being worked on'''
    #print(__file__)
    my_dir = Path(__file__).resolve().parent # go up one level to tests folder, .resolve() alone gives the absolute location
    print(my_dir)
    img_dir = my_dir.parent.joinpath('data', 'calibration', 'z1') #.parent - go up to main level (autocalib-for-mono)
    # then go into 'data/raw/z1_liver'
    print('Full path to zoom level subfolder:', img_dir)
    img_paths = img_dir.glob('*.png')  # Get all .png files in the directory
    # or img_paths = glob.glob(f'{img_dir}/*.png')  # Get all .png files as a list of strings
    #img_paths = list(img_dir.glob('*.png'))  # This will now be a list, not a map object


    for img_pth in img_paths:
    # Iterates through each file path in img_paths
        try:
            # Convert img_paths to a list
            img_paths = list(img_paths)

            if not img_paths:
                raise ValueError('No image files found in the directory.')
            
            # Use first image for processing
            first_img_pth = img_paths[4]

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
    
    save_dir = Path(__file__).resolve().parent.parent.joinpath('data', 'processed', 'z5_debugging')  # Example output folder
    # Define the full path for the output image
    output_image_path = save_dir.joinpath('calibresult-test.png')

    # Call 'Resize img' functrion
    img_resized = resize_image(img)
    cv.imshow('resized img',img_resized)
    cv.waitKey(1000)

    # Preprocess image (Gaussian blur and thresholding)
    thresh = blur_image(img_resized)
    cv.imshow('thresholded img',thresh)
    cv.waitKey(1000)

    # Find connected components
    num_labels, labels, stats, centroids = find_connected_components(thresh, connectivity=8)

    # Filter components based on size criteria
    mask, color_copy = filter_components(centroids, thresh, stats, labels, num_labels, min_width=400, min_height=400, min_area=3000)
    cv.imshow('ctd comp img', color_copy)
    cv.waitKey(1000)

    # Clean the mask using morphological operations
    mask_cleaned = clean_mask(mask)
    cv.imshow('clean mask', mask_cleaned)
    cv.waitKey(1000)

    # Apply Canny edge detection
    edges = cv.Canny(mask_cleaned, 50, 150)
    cv.imshow('canny edge detected image', edges)
    cv.waitKey(1000)

    # Detect circles in the edge-detected image
    circles = detect_circles(edges)

    # Draw circles and display the result
    color_output = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)
    color_output = draw_circles(color_output, circles)
    cv.imshow('detected circle', color_output)
    cv.waitKey(1000)

if __name__ == '__main__':
    main()