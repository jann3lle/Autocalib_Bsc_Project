import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import os

# ---- Configuration ---- #
zoom_num = 5
rows = 8
cols = 11
checker_size = 22.5 # in mm
marker_size = 16.0 # in mm

# Define image directory 
base_dir = Path(__file__).resolve().parent.parent  # Define the base directory
img_dir = base_dir / 'data' / 'calibration' / f'z{zoom_num}_images'  # Define image folder path

# Save matrices 
results_folder = base_dir / 'data' / 'calibration' / 'calibration_results' / f'z{zoom_num}_images' 
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Define ArUco dictionary (this should match the markers printed on your board)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create a virtual ChArUco board for comparison
board = aruco.CharucoBoard((cols,rows), checker_size, marker_size, aruco_dict)

# Initialize the DetectorParameters 
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Prepare arrays to store object points and image points
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane

# Load all images from the folder
images = list(img_dir.glob('*.png'))  # Change extension to match your files (.jpg, .png, etc.)
valid_images = 0 # Counter for valid images

for image_path in images:
    # Load the image
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement to help detection
    gray = cv2.equalizeHist(gray)

    # Use the ArUcoDetector object to detect markers
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:

        print(f"Detected {len(ids)} markers in {image_path.name}")
        #print(f"Marker IDs: {ids.flatten()}")

        #refine markers before interpolation

        # Interpolate ChArUco corners (refine marker corner detection)
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

        # Validate detected ChArUco corners
        if retval == 0 or charuco_corners is None or len(charuco_corners) == 0:
            print(f"No ChArUco corners detected in {image_path.name}.")
            continue

        print(f"Interpolated {len(charuco_corners)} ChArUco corners in {image_path.name}")

        # Ensure correct shape 
        if charuco_corners.shape[1] != 2:
            charuco_corners = charuco_corners.reshape(-1,1,2)

        if retval > 0 and charuco_corners is not None and len(charuco_corners) >= 4:
            print(f"Using {len(charuco_corners)} ChArUco corners from {image_path}")
            
            # Get corresponding 3D object points
            objp = board.getChessboardCorners()[charuco_ids.flatten()]
            
            # Append object points and image points
            obj_points.append(objp)
            img_points.append(charuco_corners)

            valid_images = +1

            # Draw detected markers and ChArUco corners on the image
            aruco.drawDetectedMarkers(image, corners)
            aruco.drawDetectedCornersCharuco(image, charuco_corners)
        else:
            print(f"Skipping {image_path.name}: Not enough ChArUco corners detected ({len(charuco_corners) if charuco_corners is not None else 0})")    

    else:
        print(f"No markers detected in image: {image_path.name}")
        #continue

    # Show the image with detected markers (if any)
    cv2.imshow(f"Detected ArUco Markers - {image_path.name}", image)

    key = cv2.waitKey(0)

    if key:
        cv2.destroyAllWindows()

    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Final validation before calibration
if valid_images >= 1:
    print(f"\nProceeding with calibration using {valid_images} valid images")
    # Print the length of obj_points and img_points to check if they match
    print(f"Object points: {len(obj_points)}")
    print(f"Image points: {len(img_points)}")

    # Verify number of img_points 
    # Check if the arrays match in length before proceeding with calibration
    if len(obj_points) == len(img_points):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        if ret:
            print("Calibration successful!")
            print(f"Intrinsic Matrix:\n{mtx}")
            print(f"Distortion Coefficients:\n{dist}")
            
            np.savetxt( f'{results_folder}/mtx.txt', mtx)
            np.savetxt( f'{results_folder}/dist.txt', dist)

            # Select an image for undistortion
        test_image_path = str(images[0])  # Use the first image from the dataset
        test_image = cv2.imread(test_image_path)

        # Get the image dimensions
        h, w = test_image.shape[:2]

        # Compute the new camera matrix
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort the image
        undistorted_img = cv2.undistort(test_image, mtx, dist, None, new_camera_mtx)

        # Crop the image
        #x, y, w, h = roi
        #qundistorted_img = undistorted_img[y:y+h, x:x+w]

        # Show the results
        cv2.imshow("Original Image", test_image)
        cv2.imshow("Undistorted Image", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the undistorted image (optional)
        #cv2.imwrite("undistorted_image.png", undistorted_img)

    else:
        print("Calibration failed.")

else:
    print("Error: The number of object points and image points do not match.")

