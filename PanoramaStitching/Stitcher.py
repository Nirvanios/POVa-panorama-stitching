import cv2
import numpy as np

from PanoramaStitching import Blender


def project_on_cylinder(img, center, focal):
    """
    Performs a cylindrical projection of a planar image.
    """

    if not focal:
        focal = 750

    # define mapping functions
    scale = focal
    mapX = lambda y, x: focal * np.tan(x / scale)
    mapY = lambda y, x: focal / np.cos(x / scale) * y / scale

    def make_map(y, x):
        map_x = mapX(y - center[1], x - center[0]) + center[0]
        map_y = mapY(y - center[1], x - center[0]) + center[1]
        return np.dstack((map_x, map_y)).astype(np.int16)

    # create the LUTs for x and y coordinates
    map_xy = np.fromfunction(make_map, img.shape[:2], dtype=np.int16)
    img_mapped = cv2.remap(img, map_xy, None, cv2.INTER_NEAREST)

    return img_mapped


def stitch_images(img1, img2, homography_matrix):
    """
    Stitch two images together. Applies homography_matrix on the second image
    :param img1:
    :param img2:
    :param homography_matrix:
    :return:
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    img2 = project_on_cylinder(img2, (rows2 / 2, cols2 / 2), 500)

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, homography_matrix)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    h_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, h_translation.dot(homography_matrix), (x_max - x_min, y_max - y_min))

    temp_image = np.zeros((y_max - y_min, x_max - x_min, 3), np.uint8)

    #Blender.alpha_blend(img1, output_img, translation_dist)

    temp_image[translation_dist[1]:rows1 + translation_dist[1],
    translation_dist[0]:cols1 + translation_dist[0]] = img1

    width, height, _ = temp_image.shape

    for x in range(width):
        for y in range(height):
            maxpix = temp_image[x, y].max()
            if output_img[x, y, 0] > maxpix or output_img[x, y, 1] > maxpix or output_img[x, y, 2] > maxpix:
                temp_image[x, y] = output_img[x, y]

    return temp_image
