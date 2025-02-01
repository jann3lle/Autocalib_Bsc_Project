import cv2 as cv
import numpy as np
import cv2.aruco as aruco
from pathlib import Path

# Define ChArUco board parameters
rows = 8
cols = 11
checker_size = 22.5 # in mm
marker_size = 16.0 # in mm

# Define Aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


# Initialize the DetectorParameters directly using specific board parameters
board = aruco.CharucoBoard((cols, rows), checker_size, marker_size, aruco_dict)

# Initialize the DetectorParameters directly
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Prepare arrays to store points
objpoints = []
imgpoints = []

# Load images and detect markers
base_dir = Path(__file__).resolve().parent.parent # Define image folder path
img_dir = base_dir.joinpath('data', 'calibration', 'z2_images')
images = list(img_dir.glob('*.png'))

if not img_dir.exists():
    print("The directory does not exist.")
elif not any(img_dir.iterdir()):  # Check if the directory is empty
    print("The directory is empty.")
else:
    print("The directory contains files.")

for image_path in images:
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(ids) > 0:
        # Interpolate ChArUco corners
        charuco_corners, charuco_ids, status = aruco.interpolateCornersCharuco(corners, ids, gray, board)

        if charuco_corners is not None and charuco_ids is not None:
            # Generate the object points (3D points in the real world)
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * checker_size
            objpoints.append(objp)
            imgpoints.append(charuco_corners)

            charuco_corners = np.array(charuco_corners, dtype=np.float32)

        # Display results
        aruco.drawDetectedMarkers(image, corners, ids)
        aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

    cv.imshow('Charuco Detection', image)
    cv.waitKey(1) 

cv.destroyAllWindows()       
