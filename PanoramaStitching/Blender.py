import cv2
import numpy as np


def alpha_blend(image_a, image_b, image_a_midpoint, image_b_midpoint):
    """
    Blends two images together
    :param image_a: first image (panorama)
    :param image_b: second image
    :param translation_dist:
    :return: blended image
    """
    rows, cols = image_a.shape[:2]

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    mask_a = cv2.threshold(gray_a, 1, 255, cv2.THRESH_BINARY)[1]
    mask_b = cv2.threshold(gray_b, 1, 255, cv2.THRESH_BINARY)[1]

    mask = cv2.bitwise_and(mask_a, mask_b)
    cv2.imshow('mask', mask)

    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    ext_left = tuple(contours[0][contours[0][:, :, 0].argmin()][0])[0]
    ext_right = tuple(contours[0][contours[0][:, :, 0].argmax()][0])[0]
    ext_top = tuple(contours[0][contours[0][:, :, 1].argmin()][0])[1]
    ext_bot = tuple(contours[0][contours[0][:, :, 1].argmax()][0])[1]


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.erode(mask, kernel, iterations=1)

    step = 0.0
    value = 0
    interpolation = np.zeros(image_b.shape, dtype=np.uint8)
    result = np.zeros(image_b.shape, dtype=np.uint8)
    diff = image_b_midpoint[0] - image_a_midpoint[0]
    left_right = False
    top_bot = False
    if abs(diff[0]) < abs(diff[1]):
            top_bot = True
            step = 1.0 / (ext_bot - ext_top)
    else:
        left_right = True
        step = 1.0 / (ext_right - ext_left)
    if (image_a_midpoint[0, 0] > image_b_midpoint[0, 0]) or (image_a_midpoint[0, 1] > image_b_midpoint[0, 1]):
        image_b, image_a = image_a, image_b
    for x in range(0, interpolation.shape[1]):
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

    cv2.imshow('alpha', result)
    cv2.imwrite('alpha.jpg', result)
    cv2.waitKey()

    return result
