import cv2
import numpy as np
import math

import PanoramaStitching.Utils as Utils


def alpha_blend(image_a, mask_a, image_b, mask_b, image_a_midpoint, image_b_midpoint):
    """
    Blends two images together
    :param image_a: first image (panorama)
    :param image_b: second image
    :param image_a_midpoint: midpoint of image_a
    :param image_b_midpoint: midpoint of image_b
    :return: two images blended into panorama
    """
    rows, cols = image_a.shape[:2]

    Utils.cut_pixels_around_image(image_a, mask_a)
    Utils.cut_pixels_around_image(image_b, mask_a)

    maskA = cv2.cvtColor(mask_a, cv2.COLOR_BGR2GRAY)
    maskB = cv2.cvtColor(mask_b, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(maskA, maskB)

    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    ext_left = cols
    ext_right = 0
    ext_top = rows
    ext_bot = 0
    for item in contours:
        left = tuple(item[item[:, :, 0].argmin()][0])[0]
        right = tuple(item[item[:, :, 0].argmax()][0])[0]
        top = tuple(item[item[:, :, 1].argmin()][0])[1]
        bot = tuple(item[item[:, :, 1].argmax()][0])[1]

        if left < ext_left:
            ext_left = left
        if right > ext_right:
            ext_right = right
        if top < ext_top:
            ext_top = top
        if bot > ext_bot:
            ext_bot = bot

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)

    step = 0.0
    value = 0

    result = np.zeros(image_b.shape, dtype=np.uint8)
    diff = image_b_midpoint[0] - image_a_midpoint[0]
    left_right = False
    top_bot = False
    if abs(diff[0]) < abs(diff[1]):
        top_bot = True
        step = 1.0 / abs(ext_bot - ext_top)
    else:
        left_right = True
        step = 1.0 / abs(ext_right - ext_left)

    if ((image_a_midpoint[0, 0] > image_b_midpoint[0, 0]) and left_right) \
            or ((image_a_midpoint[0, 1] > image_b_midpoint[0, 1]) and top_bot):
        image_b, image_a = image_a, image_b

    for x in range(0, image_b.shape[1]):
        for y in range(rows):
            if np.all(abs(image_a[y, x] - image_b[y, x]) > 200) and np.all(image_b[y, x] != (0, 0, 0)):
                result[y, x] = image_b[y, x]
            elif mask[y, x] == 255:
                result[y, x, :] = np.clip((value * image_a[y, x, :]) + ((1.0 - value) * image_b[y, x, :]), 0, 255)
            else:
                if np.all(image_a[y, x] > image_b[y, x]):
                    result[y, x] = image_a[y, x]
                else:
                    result[y, x] = image_b[y, x]
            if (y > ext_top) and (y < ext_bot) and top_bot:
                value += step
        if (x > ext_left) and (x < ext_right) and left_right:
            value += step
        if top_bot:
            value = 0

    return result


def simple_overlay(imageA, maskA, imageB, maskB):
    result = np.copy(imageA)
    loc = np.where(maskB == 255)
    result[loc[0], loc[1]] = imageB[loc[0], loc[1]]
    return result


def weighted_compositing(imageA, maskA, imageB, maskB, centerA, centerB):
    # Utils.cut_pixels_around_image(imageA, maskA)
    Utils.cut_pixels_around_image(imageB, maskB)
    mask_overlap = Utils.get_overlapping_mask(imageA, imageB)
    mask_overlap_blur = cv2.GaussianBlur(mask_overlap, (101, 101), cv2.BORDER_DEFAULT)

    result = np.copy(imageA)
    loc = np.where(maskB == 255)
    result[loc[0], loc[1]] = imageB[loc[0], loc[1]]

    height, width, _ = imageA.shape
    for y in range(height):
        for x in range(width):
            if mask_overlap[y, x] == 255:
                distanceA = euclidan_distance(np.array(centerA[0]), np.array((y, x)))
                distanceB = euclidan_distance(np.array(centerB[0]), np.array((y, x)))
                ratioA = distanceA / (distanceA + distanceB)
                ratioB = 1 - ratioA
                result[y, x] = imageA[y, x] * ratioA + imageB[y, x] * ratioB

    return result


def euclidan_distance(point1, point2):
    diff = abs(point1 - point2)
    tmp = diff[0] * diff[0] + diff[1] * diff[1]
    return math.sqrt(tmp)
