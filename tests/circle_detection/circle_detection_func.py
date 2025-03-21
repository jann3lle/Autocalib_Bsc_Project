import numpy as np
import cv2 as cv
import os
import glob
from pathlib import Path

def resize_image(img, width=800):
    ''' Function to resize the image'''
    height = int((width / img.shape[1] * img.shape[0]))
    img_resized = cv.resize(img, (width, height))
    return img_resized
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
        #print(f"Component {i}: Bounding Box ({x}, {y}, {w}, {h}), Area: {area}, Centroid: ({cX:.2f}, {cY:.2f})")
        
        if w > min_width and h > min_height and area > min_area:
            component_mask = (labels == i).astype("uint8") * 255
            mask = cv.bitwise_or(mask, component_mask)
            
            # Draw bounding box and centroid on the image
            cv.rectangle(color_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Bounding box
            cv.circle(color_copy, (int(cX), int(cY)), 4, (0, 0, 255), -1)  # Centroid

    cv.imshow('filtered img', color_copy)
    cv.waitKey(0)
    return mask, color_copy

def clean_mask(mask, kernel_size=(13, 13)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    return cv.morphologyEx(mask, cv.MORPH_ELLIPSE, kernel)

def detect_circles(edges, min_radius=200, max_radius=600, adjusted_dp = 1.4, adjusted_param2= 6):
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=adjusted_dp, minDist=600,
                                param2=adjusted_param2, minRadius=min_radius, maxRadius=max_radius)
    num_circles =  circles.shape[1] # Number of circles detected
    if circles is not None:
        print(f"Circles detected: {num_circles}")

        circles = np.round(circles[0, :]).astype("int")  # Convert to integers
        for (x, y, r) in circles:
            print(f"x-coordinate: {x}, y-coordinate: {y}, radius: {r}")

    else:
        print("No circles detected.")
    
    return circles







def draw_circles(img_resized, circles):
    if circles is not None:
        #circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(img_resized, (x, y), r, (255, 0, 255), 4)
            cv.circle(img_resized, (x, y), 2, (255, 0, 255), 3)
            cv.putText(img_resized, f'Radius: {r}px', (x - 40, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print('No circle detected in draw_circles function')
    cv.imshow('detected circle', img_resized)
    cv.waitKey(0)
    return img_resized, circles


