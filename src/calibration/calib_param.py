import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import os

def load_images(img_dir):
    ''' Loads all images from the directory'''
    img_paths = list(img_dir.glob("*.png"))
    valid_images = 0 # Initialise counter for valid images

    if not img_paths:
        print("No images found in the directory")
    return img_paths, valid_images

def detect_charuco_corners(img_path, aruco_detector, board):
    ''' Detects Aruco markers and interpolated Charuco corners'''
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Unable to load {img_path}")
        return None, None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print(f"No markers detected in {img_path.name}")
        return None, None, None
    
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

    if retval == 0 or charuco_corners is None or len(charuco_corners) == 0:
        print(f"No Charuco corners detected in {img_path.name}")

    # Threshold for skipping an image (to obtain mtx and dist)
    if retval == 0 or len(charuco_corners )< 6:
        print(f"Skipping {img_path.name}")
        return None, None, None
    else:
        # Draw detected markers and Charuco corners on the image
        aruco.drawDetectedMarkers(image, corners)
        aruco.drawDetectedCornersCharuco(image, charuco_corners)

    # Show image with detected markers
    cv2.imshow(f"Detected Aruco Markers - {img_path.name}: {len(charuco_corners)} corners", image)
    key = cv2.waitKey(0)

    if key:
        cv2.destroyAllWindows()

    if key == ord('q'):
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

    return charuco_corners, charuco_ids, gray

def collect_calibration_data(img_paths, valid_images):
    ''' Processes images and collects object & image points, as well as the relevant image path list'''
    # Prepare arrays to store relevant object points and image points
    obj_points, img_points = [], []
    gray = None

    for img_path in img_paths:
        charuco_corners, charuco_ids, gray_img = detect_charuco_corners(img_path, aruco_detector, board)

        if charuco_corners is not None:
                                                
            objp = board.getChessboardCorners()[charuco_ids.flatten()]
            # Add obj/img points to list, as well as relevant image path (some images = rejected due to insufficient corner detection)
            obj_points.append(objp) # Object points of charuco board in 3D
            img_points.append(charuco_corners) # Detected charuco board corners in 3D
            valid_images += 1
            gray = gray_img

    return obj_points, img_points, valid_images, gray

def save_parameters(img_paths, gray, obj_points, img_points, valid_images):
    ''' Save mtx and dist for each zoom level'''

    if valid_images >= 1:
        print(f"Proceeding with calibration using {valid_images} valid images")

        # Check if the arrays match in length before proceeding with calibration
        if len(obj_points) == len(img_points):
            ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

            if gray is None:
                print("Error: No valid images for calibration")
                return
            
            if ret:
                print("Calibration successful!")
                print(f"Intrinsic Matrix: {mtx}")
                print(f"Distortion Coefficients: {dist}")

                np.savetxt(f"{results_folder}/mtx1.txt", mtx)
                np.savetxt(f"{results_folder}/dist1.txt", dist)

                # Select an image for undistortion
                test_image_path = str(img_paths[0])
                test_image = cv2.imread(test_image_path)

                h, w = test_image.shape[:2]

                # Compute new camera matrix
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

                # Undistort image
                undistorted_img = cv2.undistort(test_image, mtx, dist, None, new_camera_mtx)

                # Show results
                cv2.imshow("Original Image", test_image)
                cv2.imshow("Undistorted Image", undistorted_img)
                cv2.waitKey(0)
            
            cv2.destroyAllWindows()

        else:
            print("Calibration Failed.")

    else:
        print("Error: The number of object points and image points do not match.")

    return 

def main():
    
    # Load all image paths
    img_paths, valid_images = load_images(img_dir)
    print(f"images found.")

    if not img_paths:
        return
    
    obj_points, img_points, valid_images, gray = collect_calibration_data(img_paths, valid_images)

    if valid_images == 0 or gray is None:
        print(" No valid images found for calibration")
        return
    
    save_parameters(img_paths, gray, obj_points, img_points, valid_images)
    
if __name__ == "__main__":

    # Configuration
    zoom_num = 1
    rows = 8
    cols = 11
    checker_size = 22.5 #mm
    marker_size = 16.0 #mm
    
    # Define image directory 
    base_dir = Path(__file__).resolve().parent.parent.parent  # Define the base directory
    img_dir = base_dir / 'data' / 'calibration' / f'z{zoom_num}_images'  # Define image folder path

    # Save matrices 
    results_folder = base_dir / 'data' / 'calibration_results' / f'z{zoom_num}_images' 
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols,rows), checker_size, marker_size, aruco_dict)

    detector_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

    main()