import os

import cv2
import glob

from PanoramaStitching.PanoramaImage import PanoramaImage


def load_images(folder):
    """
    Get list of images in a folder
    :param folder: folder containing images
    :return: list of images and their names
    """
    filenames = glob.glob(folder + "/*.png")
    images = [cv2.resize(cv2.imread(img), (480, 320)) for img in filenames]
    ret = []
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
    _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = image[y:y + h, x:x + w]
    return crop


def get_overlapping_mask(image_a, image_b):
    """
    Creates mask of overlaping region
    :param image_a: firrst image
    :param image_b: second image
    :return: mask of overlaping region
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    mask_a = cv2.threshold(gray_a, 1, 255, cv2.THRESH_BINARY)[1]
    mask_b = cv2.threshold(gray_b, 1, 255, cv2.THRESH_BINARY)[1]

    return cv2.bitwise_and(mask_a, mask_b)