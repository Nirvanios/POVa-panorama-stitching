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
