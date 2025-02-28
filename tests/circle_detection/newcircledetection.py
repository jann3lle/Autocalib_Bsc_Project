# Import the necessary packages
import numpy as np
import cv2 
import os
import glob
from pathlib import Path



def get_circle_radius(img, waitTime=1):
    connectivity = 8

    # Resize the image to fit the screen (adjust size as needed)
    width = 800  # Set the width for resizing (change to suit your needs)
    height = int((width / img.shape[1]) * img.shape[0])  # Maintain aspect ratio
    img_resized = cv2.resize(img, (width, height))  # Resize the image

    # Apply simple thresholding + gaussian filter (Blurred image)
    blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)

    # Apply connected component analysis
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    # Initialise output mask to store all characters parsed from license plate
    mask = np.zeros(img_resized.shape, dtype="uint8")

    # Create a copy of the resized image once, outside the loop
    output = img_resized.copy()

    # Convert to BGR to draw green bounding box
    color_output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # Output folder
    output_folder = 'single_circle_output_images'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop over the number of unique connected component labels
    for i in range(1, numLabels):
        # Extract the connected component statistics and centroid for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        print(f"Component {i}: Bounding Box ({x}, {y}, {w}, {h}), Area: {area}, Centroid: ({cX:.2f}, {cY:.2f})")

        # Filter connected components
        keepWidth =  w > 400
        keepHeight = h > 400
        keepArea = area > 3000

        # Ensure the connected component we are examining passes all three tests
        if all((keepWidth, keepHeight, keepArea)):
            print(f"[INFO] keeping connected component '{i}'")
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

            print(f"Image dimensions:", color_output.shape)  # Check image dimensions
            print(f"Connected component statistics: x: {x}, y: {y}, w: {w}, h: {h}")
            cv2.rectangle(color_output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print("Drawing rectangle...")
            cv2.circle(color_output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        else:
            print(f"[INFO] Discarded component '{i}' - Filters not met.")

    # Morphological operations to remove black regions in white circle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask_cleaned = cv2.morphologyEx(componentMask, cv2.MORPH_CLOSE, kernel)

    # Apply Canny Edge Detection (on mask_cleaned image)
    # Don't see any notable changes when changing the thresholds ****
    low_threshold = 50  # Lower bound of gradient intensity
    high_threshold = 150 # Upper bound of gradient intensity

    # Perform Canny edge detection
    edges = cv2.Canny(mask_cleaned, low_threshold, high_threshold)

    # Detect circles using Hough Circle Transform on Canny edge image
    # only param2 and dp relevant
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=600,
                                param2=30, minRadius=200, maxRadius=600)

    # If some circles are detected
    if circles is not None:
        # Convert the coordinates and radius of the circles into integers
        circles = np.round(circles[0, :]).astype("int")
            
        # Iterate through each circle detected
        for (x, y, r) in circles:
                # Draw the circle in the original image
            cv2.circle(color_output, (x, y), r, (255, 0, 255), 4)
                
                # Draw the center of the circle
            cv2.circle(color_output, (x, y), 2, (255, 0, 255), 3)
                
                # Print the radius of the circle
            print(f"Radius of circle at ({x}, {y}): {r} pixels")

                # Add radius text to image
            cv2.putText(color_output, f'Radius: {r}px', (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("No circles were detected in the image.")

    # Show our output image and connected component mask
    #cv2.imshow("Original image", color_output) # Highlights connected component 
    #cv2.imshow("Connected Component", mask_cleaned) # Displays the connected component region before Filtering
    #cv2.imshow("Canny Edge Detection", edges)

    if circles is not None:
        cv2.imshow('Detected Circles with Detected circle', color_output)  # Display the image in an OpenCV window
    else:
        print (f"No circles detected")

    #cv2.imwrite(os.path.join(output_folder, "z5.png"), color_output)
    #cv2.imwrite(os.path.join(output_folder, "zoom4_22x22.png"), mask_cleaned)
    #cv2.imwrite(os.path.join(output_folder, "Canny-edge(5).png"), edges)
    cv2.waitKey(waitTime)


    return r



def main():

    

    current_dir = Path(__file__).resolve()
    pth = current_dir.joinpath( 'zoom_1_liver_frames')
    # Define the path to the image and connectivity value
    img_pth_lst = glob.glob(f'{pth}/*.png')
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        cv2.imshow('images',img)
        cv2.waitKey(1)
        if cv2.waitKey(1)==ord('q'):
            break

    for img_pth in img_pth_lst:
        img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "File could not be read, check with os.path.exists()"

        radius = get_circle_radius(img)
        print(radius)
    
    cv2.destroyAllWindows()
    return

if __name__=='__main__':
    main()
