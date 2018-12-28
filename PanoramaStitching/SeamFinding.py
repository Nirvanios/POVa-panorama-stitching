import heapq
from collections import namedtuple

import cv2
import numpy as np
import networkx as nx

from PanoramaStitching import Utils


def get_inverted_difference_image(image_a, image_b):
    """
    Creates and returns inverted difference of two images
    :param image_a: First image
    :param image_b: Second image
    :return: inverted difference of images
    """
    diff = image_a - image_b
    return 255 - diff


def get_3x3_neighbour_elements(matrix, x, y):
    """
    Returns 3x3 matrix around given coordinates
    :param matrix: Matrix in which to search
    :param x: X coordination
    :param y: Y coordination
    :return: 3x3 neighbour matrix
    """
    width, height = matrix.shape[:2]
    neighbour = np.ones((3, 3), dtype=int)
    neighbour = np.negative(neighbour)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (width >= x + i >= 0) and (height >= y + j >= 0):
                neighbour[i + 1, j + 1] = matrix[x + i, y + j]
    return neighbour


def get_4neighbour_elements(matrix, x, y):
    """
    Returns 4 elements (cross) around given coordinates
    :param matrix: Matrix in which to search
    :param x: X coordination
    :param y: Y coordination
    :return: 4 elements (cross) around given coordinates
    """
    width, height = matrix.shape[:2]
    neighbours = np.ones(4)
    neighbours = np.negative(neighbours)
    if x - 1 > 0:
        neighbours[0] = matrix[x - 1, y]
    if x + 1 < width:
        neighbours[1] = matrix[x + 1, y]
    if y - 1 > 0:
        neighbours[2] = matrix[x, y - 1]
    if y + 1 < height:
        neighbours[3] = matrix[x, y + 1]
    return neighbours


def get_watershed_image(image, mask):
    """
    Creates and returns segmented image.
    In returned image left segment has value -5 and right segment -10.
    Those two segments represents source and sink in graph.
    :param image: inverted difference of two images
    :param mask: mask of overlapping region
    :return: Segmented image
    """
    height, width = mask.shape
    original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    original_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    box_img = cv2.cornerHarris(mask, 5, 3, 0.04)
    """
    ret, box_img = cv2.threshold(box_img, 0.1 * box_img.max(), 255, 0)
    dst = np.uint8(box_img)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (5, 5), (-1, -1), criteria)
    corners = corners.astype(int)
        """
    flat = box_img.flatten()
    indices = np.argpartition(flat, -4)[-4:]
    indices = indices[np.argsort(-flat[indices])]
    corners = np.unravel_index(indices, box_img.shape)
    corners = corners[1]
    ext_left = (np.sort(corners))[0]
    ext_right = (np.sort(corners))[-1]
    ext_left2 = (np.sort(corners))[1]
    ext_right2 = (np.sort(corners))[-2]

    gray = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = np.where(mask == 0, mask, gray)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=2)
    half = int(ext_left + ((ext_right - ext_left) / 2))
    for i in range(height):
        if mask[i, half] != 255:
            gray[i, half] = 255
    # cv2.imshow('gray2', gray)
    # cv2.waitKey()

    ret, markers = cv2.connectedComponents(gray)

    watershed_image = cv2.watershed(original_image, markers)
    original_mask[watershed_image == -1] = [0, 0, 255]

    temp_labels = np.unique(watershed_image[:, 1])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = -5

    temp_labels = np.unique(watershed_image[:, width - 2])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = -10

    stepped_over = np.zeros(height, dtype=bool)
    for x in range(1, ext_right):
        for y in range(height):
            if watershed_image[y, x] == -1:
                stepped_over[y] = True
            if ((mask[y, x] == 0 and x < ext_left2 + 1) or (not stepped_over[y] and mask[y, x] == 255)) and watershed_image[
                y, x] != -1:
                watershed_image[y, x] = -5

    new_watershed = watershed_image
    iteration = 0
    changed = True
    original_mask[watershed_image == -5] = [255, 0, 0]
    original_mask[watershed_image == -10] = [0, 255, 0]
    # cv2.imshow('water_b', original_mask)
    # cv2.waitKey()
    while changed:
        changed = False
        iteration += 1
        for (x, y), item in np.ndenumerate(watershed_image):
            if item == -10 and mask[x, y] == 255 and watershed_image[x, y] != -1:
                neighbours = get_4neighbour_elements(new_watershed, x, y)
                if np.any(neighbours == -5):
                    new_watershed[x, y] = -5
                    changed = True
        watershed_image = new_watershed

    watershed_image[watershed_image[:, ext_right - 1] == -5, ext_right - 1] = -1

    stepped_over = np.zeros(height, dtype=bool)
    for x in range(2, (width - ext_left)):
        for y in range(height):
            if watershed_image[y, -x] == -1:
                stepped_over[y] = True
            if -x > -(width - ext_right2):
                stepped_over[mask[:, x] == 0] = True
            if ((mask[y, -x] == 0 and -x > -(width - ext_right2)) or (not stepped_over[y] and mask[y, -x] == 255)) and watershed_image[
                y, -x] != -1:
                watershed_image[y, -x] = -4

    new_watershed = watershed_image
    iteration = 0
    changed = True
    """
    original_mask[watershed_image == -4] = [255, 0, 255]
    original_mask[watershed_image == -5] = [255, 0, 0]
    original_mask[watershed_image == -10] = [0, 255, 0]
    cv2.imshow('water_m', original_mask)
    cv2.waitKey()
    """
    while changed:
        changed = False
        iteration += 1
        for (x, y), item in np.ndenumerate(watershed_image):
            if (item == -10) and mask[x, y] == 255 and watershed_image[x, y] != -1:
                neighbours = get_4neighbour_elements(new_watershed, x, y)
                if np.any(neighbours == -4):
                    new_watershed[x, y] = -4
                    changed = True
        watershed_image = new_watershed

    """
    original_mask[watershed_image == -4] = [255, 0, 255]
    original_mask[watershed_image == -5] = [255, 0, 0]
    original_mask[watershed_image == -10] = [0, 255, 0]

    cv2.imshow('water?', original_mask)
    """
    watershed_image[watershed_image == -10] = -6
    watershed_image[watershed_image == -4] = -10

    original_mask[watershed_image == -5] = [255, 0, 0]
    original_mask[watershed_image == -10] = [0, 255, 0]
    original_mask[watershed_image == -6] = [255, 0, 255]
    original_mask[watershed_image == -1] = [0, 0, 255]

    # cv2.imshow('water', original_mask)
    # cv2.waitKey()

    return watershed_image


def find_cut_seam(segmented_image, inverted_image, mask):
    """
    Creates graph and perform minimal cut on it.
    :param segmented_image: segmented image from get_watershed_image function
    :param inverted_image: inverted difference of two images
    :param mask: mask of overlapping region
    :return: List of node pairs where is minimal cut found
    """
    original_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    height, width = segmented_image.shape
    Edge = namedtuple("Edge", ["node_1", "node_2"])
    edges = {}
    graph = nx.Graph()
    for x in range(height):
        for y in range(width):
            if mask[x, y] == 255 and segmented_image[x, y] == -1:
                tmp_edge = Edge(node_1=-5, node_2=-5)
                tmp_edge_inv = Edge(node_1=-5, node_2=-5)
                up = segmented_image[x - 1, y]
                down = segmented_image[x + 1, y]
                left = segmented_image[x, y - 1]
                right = segmented_image[x, y + 1]
                if left != down and (left != -1 and right != -1):
                    if segmented_image[x - 1, y] == -5:
                        left = -1
                    elif segmented_image[x + 1, y] == -10:
                        right = -2
                    if left == -1 and right == -2:
                        print()
                    tmp_edge = Edge(node_1=left, node_2=right)
                    tmp_edge_inv = Edge(node_1=right, node_2=left)
                elif up != down and (up != -1 and down != -1):
                    if segmented_image[x - 1, y] == -5:
                        up = -1
                    elif segmented_image[x + 1, y] == -10:
                        down = -2
                    if up == -1 and down == -2:
                        print()
                    tmp_edge = Edge(node_1=up, node_2=down)
                    tmp_edge_inv = Edge(node_1=down, node_2=up)
                if tmp_edge == tmp_edge_inv:
                    continue

                if tmp_edge in edges:
                    edges[tmp_edge] += inverted_image[x, y]
                elif tmp_edge_inv in edges:
                    edges[tmp_edge_inv] += inverted_image[x, y]
                else:
                    edges[tmp_edge] = inverted_image[x, y].astype(int)

    for edge in edges:
        graph.add_edge(edge.node_1, edge.node_2, capacity=edges[edge])
    cut_nodes = nx.minimum_edge_cut(graph, -1, -2)

    return cut_nodes


def graph_cut_blend(image_a, image_b):
    """
    Blends two images using watershed and graph cut.
    :param image_a: First image
    :param image_b: Second image
    :return: Panorama blended using graph cut
    """
    height, width = image_a.shape[:2]
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    difference_image = get_inverted_difference_image(gray_a, gray_b)
    mask = Utils.get_overlapping_mask(image_a, image_b)
    difference_image = np.where(mask == 255, difference_image, 0)
    # cv2.imshow('difference', difference_image)
    # cv2.waitKey()
    segmented_image = get_watershed_image(difference_image, mask)
    cut_nodes = np.array(list(find_cut_seam(segmented_image, difference_image, mask)))
    cut_nodes[cut_nodes == -2] = -10
    cut_nodes[cut_nodes == -1] = -5
    new_image_a = np.copy(image_a)
    copy_black = False
    current_line = 0
    for (x, y), item in np.ndenumerate(segmented_image):
        if current_line != x:
            copy_black = False
        if x == 0 or y == 0 or x == height - 1 or y == width - 1:
            continue
        left = segmented_image[x, y - 1]
        right = segmented_image[x, y + 1]
        rightmost = right
        if segmented_image[x, y] == -1:
            if copy_black:
                i = 1
                while rightmost == -1 and y + i < width - 1:
                    i += 1
                    rightmost = segmented_image[x, y + i]
                if y + i >= width - 1:
                    copy_black = True
                    current_line = x
                    continue
            i = 1
            if left == right:
                if not copy_black:
                        new_image_a[x, y] = image_b[x, y]
                continue

            while left == -1 and y - i > 0:
                i += 1
                left = segmented_image[x, y - i]
            if y - i <= 0 and right != -1:
                copy_black = True
                current_line = x
                continue


        if segmented_image[x, y] == -1 \
                and left != right \
                and (any((cut_nodes[:] == [right, left]).all(1))
                     or any((cut_nodes[:] == [left, right]).all(1))):

            if copy_black:
                copy_black = False
            else:
                copy_black = True
            current_line = x
        if not copy_black:
                new_image_a[x, y] = image_b[x, y]

    """
    cv2.imshow('new_im', new_image_a)
    cv2.waitKey()
    """

    return new_image_a
