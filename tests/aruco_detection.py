import cv2
import cv2.aruco as aruco
import pathlib

# Define ArUco dictionary (this should match the markers printed on your board)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Initialize the DetectorParameters directly
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Directory containing images
base_dir = pathlib.Path(__file__).resolve().parent.parent  # Define the base directory
img_dir = base_dir / 'data' / 'calibration' / 'z5_images'  # Define image folder path

# Load all images from the folder
images = list(img_dir.glob('*.png'))  # Change extension to match your files (.jpg, .png, etc.)

for image_path in images:
    # Load the image
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the ArUcoDetector object to detect markers
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Draw detected markers on the image
        aruco.drawDetectedMarkers(image, corners, ids)
        print(f"Detected markers in image: {image_path.name}")

    # Show the image with detected markers (if any)
    cv2.imshow(f"Detected ArUco Markers - {image_path.name}", image)
    cv2.waitKey(500)  # Display each image for 500ms
    cv2.destroyAllWindows()

