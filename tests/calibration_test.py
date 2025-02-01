import numpy as np
import cv2 as cv
from pathlib import Path
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Image folder path
my_dir = Path(__file__).resolve().parent
img_dir = my_dir.parent.joinpath('data', 'calibration', 'z2_images')
#images = glob.glob(str(img_dir) + '*.jpg')
images = list(img_dir.glob('*.png'))

# Check if the directory exists and contains any files
if not img_dir.exists():
    print("The directory does not exist.")
elif not any(img_dir.iterdir()):  # Check if the directory is empty
    print("The directory is empty.")
else:
    print("The directory contains files.")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

# If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Combine the path to the image
img_path = img_dir.joinpath('00001.png')

# Select specific img (left12.jpg)
img = cv.imread(img_path)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Save the image to a specific location
# Define the directory where the image should be saved
save_dir = Path(__file__).resolve().parent.parent.joinpath('data', 'processed','tests')  # Example output folder
# Define the full path for the output image
output_image_path = save_dir.joinpath('calibresult-test3.png')
# Save image
if dst is None:
    print("Failed to generate the image (dst is None).")
else:
    # Proceed with saving the image
    cv.imwrite(str(output_image_path), dst)
    print(f"Image saved to: {output_image_path}")