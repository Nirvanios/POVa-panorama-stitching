import os
import cv2
import numpy as np
import glob

from PanoramaStitching.PanoramaImage import PanoramaImage


def load_images(folder):
    """
    Get list of images in a folder
    :param folder: folder containing images
    :return: list of images and their names
    """

    # Load all .png files in folder
    filenames = glob.glob(folder + "/*.png")
    images = [cv2.resize(cv2.imread(img), (480, 320)) for img in filenames]
    ret = []

    # Add images and file names into list
    for i in range(len(filenames)):
        pan_image = PanoramaImage(os.path.basename(filenames[i]), images[i])
        ret.append(pan_image)
    return ret


def crop_black(image):
    """
    Crop black parts of the image
    :param image: input image
    :return: cropped image
    """
    gr_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gr_image > 0

    coords = np.argwhere(mask)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    cropped = image[x0:x1, y0:y1]
    return cropped


def get_overlapping_mask(image_a, image_b):
    """
    Creates mask of overlapping region
    :param image_a: first image
    :param image_b: second image
    :return: mask of overlapping region
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    mask_a = cv2.threshold(gray_a, 0, 255, cv2.THRESH_BINARY)[1]
    mask_b = cv2.threshold(gray_b, 0, 255, cv2.THRESH_BINARY)[1]

    return cv2.bitwise_and(mask_a, mask_b)


def cut_pixels_around_image(image, mask, thickness=2):
    tmp_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(tmp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    for i in range(len(contours)):
        cv2.drawContours(image, contours, i, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_8)
        cv2.drawContours(mask, contours, i, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_8)
