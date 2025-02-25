import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

# --------------------- CONFIGURATION ---------------------
ROWS = 8
COLS = 11
CHECKER_SIZE = 22.5  # mm
MARKER_SIZE = 16.0  # mm

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
BOARD = aruco.CharucoBoard((COLS, ROWS), CHECKER_SIZE, MARKER_SIZE, ARUCO_DICT)

DETECTOR_PARAMS = aruco.DetectorParameters()
ARUCO_DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

# --------------------- FUNCTIONS ---------------------
def load_images(img_dir):
    """Loads all images from the directory."""
    img_paths = list(img_dir.glob("*.png"))  # Change extension if needed
    if not img_paths:
        print(" No images found in directory!")
    return img_paths

def detect_charuco_corners(image_path):
    """Detects ArUco markers and interpolates ChArUco corners."""
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Contrast enhancement

    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print(f" No markers detected in {image_path.name}")
        return None, None

    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)

    if retval == 0 or charuco_corners is None or len(charuco_corners) < 4:
        print(f" Skipping {image_path.name}: Not enough ChArUco corners ({len(charuco_corners) if charuco_corners is not None else 0})")
        return None, None

    print(f"{image_path.name}: {len(charuco_corners)} ChArUco corners detected")

    return charuco_corners, charuco_ids

def collect_calibration_data(images):
    """Processes images and collects object & image points for calibration."""
    obj_points, img_points = [], []

    for img_path in images:
        charuco_corners, charuco_ids = detect_charuco_corners(img_path)

        if charuco_corners is not None:
            objp = BOARD.getChessboardCorners()[charuco_ids.flatten()]
            obj_points.append(objp)
            img_points.append(charuco_corners)

    return obj_points, img_points

def calibrate_camera(obj_points, img_points, img_size):
    """Performs camera calibration using detected object and image points."""
    if len(obj_points) < 1:
        print("\n Not enough valid images for calibration.")
        return None, None

    print(f"\n Calibrating with {len(obj_points)} valid images...")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    if ret:
        print(" Calibration Successful!")
        print(f" Intrinsic Matrix:\n{mtx}")
        print(f" Distortion Coefficients:\n{dist}")
        #print(f" Object Points:\n{obj_points}")
        #print(f" Image Points:\n{img_points}")
        return mtx, dist
    else:
        print("\n Calibration failed.")
        return None, None

def undistort_image(image_path, mtx, dist):
    """Undistorts an image using the computed camera parameters."""
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, mtx, dist, None, new_camera_mtx)

    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------- MAIN FUNCTION ---------------------
def main():
    """Main function to handle the calibration pipeline."""
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "data" / "calibration" / "z1_images"

    images = load_images(img_dir)
    if not images:
        return

    obj_points, img_points = collect_calibration_data(images)

    if obj_points and img_points:
        mtx, dist = calibrate_camera(obj_points, img_points, (images[0].stat().st_size, images[0].stat().st_size))
        
        if mtx is not None and dist is not None:
            undistort_image(images[0], mtx, dist)  # Select an image for undistortion

if __name__ == "__main__":
    main()
