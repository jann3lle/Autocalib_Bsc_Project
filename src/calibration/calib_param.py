import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import os

def load_images(img_dir):
    ''' Loads all images from the directory'''
    img_paths = list(img_dir.glob("*.png"))
    if not img_paths:
        print("No images found in the directory")
    return img_paths

def main():
    # Load all image paths
    img_paths = load_images(img_dir)
    print("images found")

    if not img_paths:
        return

if __name__ == "__main__":

    # Configuration
    zoom_num = 2
    rows = 8
    cols = 11
    checker_size = 22.5 #mm
    marker_size = 16.0 #mm
    
    # Define image directory 
    base_dir = Path(__file__).resolve().parent.parent  # Define the base directory
    img_dir = base_dir / 'data' / 'calibration' / f'z{zoom_num}_frames'  # Define image folder path

    # Save matrices 
    results_folder = base_dir / 'data' / 'calibration_results' / f'z{zoom_num}_images' 
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols,rows), checker_size, marker_size, aruco_dict)

    detector_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # mtx and dist 

    #mtx = np.loadtxt(f'{results_folder}/mtx.txt')

    #dist = np.loadtxt(f'{results_folder}/dist.txt')
    
    main()