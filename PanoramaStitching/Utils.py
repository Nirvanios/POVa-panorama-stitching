import os
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

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


def balance_color(images, main_image):
    """
    Balance color of images using k-means clustering
    :param images: images list
    :param main_image: main image of the panorama stitching
    :return: changed images
    """
    # Convert to RGB form and float type
    image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
    img = np.float64(image) / 255

    # Get shape of the image
    w, h, d = tuple(img.shape)
    assert d == 3

    # Reduce one dimension
    img_arr = np.reshape(img, (w * h, d))

    # Use only part od random intensities in image
    img_arr_sample = shuffle(img_arr, random_state=0)[:1000]

    # Train
    kmeans = KMeans(n_clusters=64, random_state=0).fit(img_arr_sample)

    # Do it for all pictures
    for image in images:
        img = cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB)
        img = np.float64(img) / 255
        img_arr = np.reshape(img, (w * h, d))

        # Prediction based on trained model
        labels = kmeans.predict(img_arr)

        img = np.zeros((w, h, kmeans.cluster_centers_.shape[1]))
        label_id = 0

        # Iterate
        for i in range(w):
            for j in range(h):
                # Set pixel intensity
                img[i][j] = kmeans.cluster_centers_[labels[label_id]]
                label_id += 1

        # Convert from double to uint8
        img = img * 255

        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image.image = img
    return images


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
