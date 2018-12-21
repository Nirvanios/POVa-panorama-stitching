import string
import sys
import cv2
import numpy
import argparse

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')



def main():
    print("Hello world")
    print("dzea")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main()
