import string
import sys
import cv2
import numpy
import argparse

sift = cv2.xfeatures2d.SIFT_create()

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')


def get_sift(image):
    """
    Get key points detected by SIFT
    """
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points = sift.detect(gray_scale_image, None)
    return key_points


def main():
    print("Hello world")
    print("dzea")


if __name__ == "__main__":
    args = parser.parse_args()
    main()
