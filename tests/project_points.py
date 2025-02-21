import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

# --- configuartion --- #
rows = 8
cols = 11
checker_size = 22.5 #mm
marker_size = 16.0 #mm

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard((cols,rows), checker_size, marker_size, aruco_dict)

detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# mtx and dist 

mtx = np.array([
    [2.27618495e+05, 0.00000000e+00, 2.33722001e+05],
    [0.00000000e+00, 1.01587727e+06, 2.33722000e+05],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
 ])
dist = np.array([[4.98197567e+00, -9.11929155e+00, -3.20344787e-03, -1.42911756e-02, 4.13836317e+00]])


# --- functions --- #

def load_images(img_dir):
    """Loads all images from the directory."""
    img_paths = list(img_dir.glob("*.png"))  # Change extension if needed
    if not img_paths:
        print(" No images found in directory!")
    return img_paths

def detect_charuco_corners(img_path):
    ''' Detects Aruco markers and interpolated ChArUco corners.'''
    image = cv2.imread(str(img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print (f" No markers detected in {img_path.name}")
        return None, None
    
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

    if retval == 0 or charuco_corners is None or len(charuco_corners) < 4:
        print(f"Skipping {img_path.name}")
        return None, None
    
    print(f"{img_path.name}: {len(charuco_corners)}")

    return charuco_corners, charuco_ids

def collect_calibration_data(img_paths):
    ''' Processes images and collects object and image points for calibration.'''
    obj_points, img_points = [], []

    for img_path in img_paths:
        charuco_corners, charuco_ids = detect_charuco_corners(img_path)

        if charuco_corners is not None:
            objp = board.getChessboardCorners()[charuco_ids.flatten()]
            obj_points.append(objp) # Object points from 3D
            img_points.append(charuco_corners)

    return obj_points, img_points

def reprojection_error(imgpoints_detected, imgpoints_reprojected, img_path, IDs=None):
    """
    calculate reprojection error given the detected and reprojected points
    """
    diff = (imgpoints_detected - imgpoints_reprojected)
    if np.max(diff) > 1000:
        # to avoid overflow
        error_np = np.inf
    else:
        squared_diffs = np.square(diff)
        error_np = np.sqrt(np.sum(squared_diffs) / len(imgpoints_reprojected))
        # round up to 5 decimal places
        error_np = round(error_np, 5)

    if img_path is not None:
        if isinstance(img_path, Path):
            img_path = cv2.imread(str(img_path))

        img_shape = img_path.shape
        for idx, corner_detected in enumerate(imgpoints_detected):
            # change dtypw of corner to int
            corner_detected = corner_detected.astype(int)
            corner_reprojected = imgpoints_reprojected[idx].astype(int)

            centre_detected = corner_detected.ravel()
            centre_reprojected = corner_reprojected.ravel()

            # check if points are within image
            if centre_detected[0] < 0 or centre_detected[0] > img_shape[1] or centre_detected[1] < 0 or centre_detected[
                1] > img_shape[0]:
                continue
            if centre_reprojected[0] < 0 or centre_reprojected[0] > img_shape[1] or centre_reprojected[1] < 0 or \
                    centre_reprojected[1] > img_shape[0]:
                continue
            cv2.circle(img_path, (int(centre_detected[0]), int(centre_detected[1])), 3, (0, 0, 255), -1)
            cv2.circle(img_path, (int(centre_reprojected[0]), int(centre_reprojected[1])), 3, (0, 255, 0), -1)
            # TODO ADD IDs to each tag
            if IDs is not None:
                # add ID of detected tag
                ID=IDs[idx]
                cv2.putText(img_path, f'{ID}', (int(centre_detected[0]), int(centre_detected[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img_path, f'{ID}', (int(centre_reprojected[0]), int(centre_reprojected[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        return error_np, img_path
    
    return error_np

def calculate_reprojection_error(mtx, dist, objPoints, imgPoints, img_paths, waitTime=0, IDs=None):
    """
    calculate reprojection error on a set of points from images given the intrinsics and distortion coefficients

    Parameters
    ----------
    mtx : ndarray
    camera intrinsics matrix
    dist : ndarray
    distortion coefficients
    objPoints : ndarray
    3D points of the chessboard
    imgPoints : ndarray
    2D points of the chessboard

    image_pths : list of strings
    list of image paths to display the reprojection error, by default None
    waitTime : int, optional
    time to wait for key press to continue, by default 1
    """

    mean_errors = []


    for i in range(len(objPoints)):

        if len(objPoints[i]) < 4 or len(imgPoints[i]) < 4:
            continue

        # Estimate rvec and tvec using solvePnP
        retval, rvec, tvec = cv2.solvePnP(objPoints[i], imgPoints[i], mtx, dist)
        # Project 3D points to image plane
        imgpoints_reprojected, _ = cv2.projectPoints(objPoints[i], rvec, tvec, mtx, dist)
        imgpoints_detected = imgPoints[i]
        # calculate error
        if IDs is None:
                ID = None
        else:
            ID = IDs[i]
        if img_paths is not None:
            img_path = cv2.imread(str(img_paths[i]))
            
            error_np, annotated_image = reprojection_error(imgpoints_detected, imgpoints_reprojected, img_path, IDs=ID)
            cv2.imshow('charuco board', annotated_image)
            cv2.waitKey(waitTime)
        else:
            error_np = reprojection_error(imgpoints_detected, imgpoints_reprojected, IDs=ID)
        mean_errors.append(error_np)
    return mean_errors

def main():
    # Directory containing images
    base_dir = Path(__file__).resolve().parent.parent  # Define the base directory
    img_dir = base_dir / 'data' / 'calibration' / 'z5_images'  # Define image folder path

    # Load all images from the folder
    img_paths = load_images(img_dir)
    if not img_paths:
        return
    
    obj_points, img_points = collect_calibration_data(img_paths)
   
    # Calculate reprojection error for all images
    mean_errors = calculate_reprojection_error(mtx, dist, obj_points, img_points, img_paths)

    print("Mean reprojection errors for each image:", mean_errors)

if __name__ == "__main__":
    main()