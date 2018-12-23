import cv2
import numpy as np


def alpha_blend(image_a, image_b, translation_dist):
    """
    Blends two images together
    :param image_a: first image
    :param image_b: second image
    :param translation_dist:
    :return: blended image
    """
    rows, cols = image_a.shape[:2]

    black_a = np.zeros(image_b.shape, dtype=np.uint8)
    black_a[translation_dist[1]:rows + translation_dist[1],
    translation_dist[0]:cols + translation_dist[0]] = image_a
    gray_a = cv2.cvtColor(black_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    mask_a = cv2.threshold(gray_a, 1, 255, cv2.THRESH_BINARY)[1]
    mask_b = cv2.threshold(gray_b, 1, 255, cv2.THRESH_BINARY)[1]

    mask = cv2.bitwise_and(mask_a, mask_b)

    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    ext_left = tuple(contours[0][contours[0][:, :, 0].argmin()][0])[0]
    ext_right = tuple(contours[0][contours[0][:, :, 0].argmax()][0])[0]

    step = 1.0 / (ext_right - ext_left)
    value = 0
    interpolation = np.zeros(image_b.shape, dtype=np.uint8)
    result = np.zeros(image_b.shape, dtype=np.uint8)
    for x in range(0, interpolation.shape[1]):
        result[:, x, :] = np.clip((value * black_a[:, x, :]) + ((1.0 - value) * image_b[:, x, :]), 0, 255)
        if (x > ext_left) and (x < ext_right):
            value += step

    return result
