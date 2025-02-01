import numpy as np
import cv2 as cv
from pathlib import Path

# Termination criteria for cornerSubPix
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
OBJP = np.zeros((6 * 7, 3), np.float32)
OBJP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)


def load_images(image_dir):
    """
    Load image paths from the given directory.
    :param image_dir: Path to the folder containing chessboard images.
    :return: List of image file paths.
    """
    img_dir = Path(image_dir)
    
    if not img_dir.exists():
        print("The directory does not exist.")
        return []

    images = list(img_dir.glob('*.jpg'))
    
    if not images:
        print("The directory is empty.")
        return []
    
    print(f"Found {len(images)} images in {img_dir}.")
    return images


def find_chessboard_corners(images):
    """
    Find chessboard corners in the provided images.
    :param images: List of image file paths.
    :return: Tuple (object points, image points) for calibration.
    """
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    gray = None

    for fname in images:
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        if ret:
            objpoints.append(OBJP)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow('Chessboard Corners', img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    return objpoints, imgpoints, gray.shape if gray is not None else None


def calibrate_camera(objpoints, imgpoints, image_shape):
    """
    Calibrate the camera using object points and image points.
    :param objpoints: 3D real-world points.
    :param imgpoints: 2D image plane points.
    :param image_shape: Shape of the image used for calibration.
    :return: Camera matrix, distortion coefficients, rotation vectors, translation vectors.
    """
    if image_shape is None:
        print("No valid images provided for calibration.")
        return None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )
    
    if ret:
        print("Camera calibration successful!")
    else:
        print("Camera calibration failed!")

    return ret, mtx, dist, rvecs, tvecs


def undistort_image(img_path, mtx, dist, save_dir):
    """
    Undistort a given image and save the corrected version.
    :param img_path: Path to the image file to be undistorted.
    :param mtx: Camera matrix.
    :param dist: Distortion coefficients.
    :param save_dir: Directory to save the undistorted image.
    """
    img = cv.imread(str(img_path))
    h, w = img.shape[:2]

    # Get optimal new camera matrix
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Save the image
    save_path = Path(save_dir).joinpath('calibresult-test.png')
    
    if dst is None:
        print("Failed to generate the image (dst is None).")
    else:
        cv.imwrite(str(save_path), dst)
        print(f"Image saved to: {save_path}")


def main():
    # Define image folder path
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir.joinpath('data', 'calibration', 'z2_images')

    # Load images
    images = load_images(img_dir)
    if not images:
        return

    # Find chessboard corners
    objpoints, imgpoints, image_shape = find_chessboard_corners(images)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_shape)

    if ret:
        # Select a specific image for undistortion
        img_path = img_dir.joinpath('00002.png')
        if img_path.exists():
            undistort_image(img_path, mtx, dist, img_dir)
        else:
            print(f"Image {img_path} not found for undistortion.")


if __name__ == '__main__':
    main()
