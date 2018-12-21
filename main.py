import string
import sys
import cv2
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')

def load_images(folder):
    filenames = glob.glob(folder + "/*.png")
    images = [cv2.imread(img) for img in filenames]
    ret = []
    for i in range(len(filenames)):
        ret.append([images[i], os.path.basename(filenames[i])])
    return ret

def main():
    print("Hello world")
    print("dzea")
    images = load_images(args.folder)
    print(args)

if __name__ == "__main__":
    args = parser.parse_args()
    main()
