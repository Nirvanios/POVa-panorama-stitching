import os

import cv2
import glob

from src.PanoramaImage import PanoramaImage


def load_images(folder):
    """
    Get list of images in a folder
    :param folder: folder containing images
    :return: list of images and their names
    """
    filenames = glob.glob(folder + "/*.png")
    images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in filenames]
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
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = image[y:y + h, x:x + w]
    return crop
