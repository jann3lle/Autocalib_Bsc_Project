import numpy as np
import cv2 as cv
import os
import glob
from pathlib import Path

print('hi')

def main():
    ''' Where __file__ rep the current file being worked on'''
    #print(__file__)
    my_dir = Path(__file__).resolve().parent # go up one level to tests folder, .resolve() alone gives the absolute location
    print(my_dir)
    img_dir = my_dir.parent.joinpath('data', 'raw', 'z1_white') #.parent - go up to main level (autocalib-for-mono)
    # then go into 'data/raw/z1_white'
    print('Full path to z1 white:', img_dir)
    img_paths = img_dir.glob('*.png')  # Get all .png files in the directory
    #img_paths = glob.glob(f'{img_dir}/*.png')  # Get all .png files as a list of strings
    #print(img_paths)

if __name__ == '__main__':
    main()

cv.destroyAllWindows()

