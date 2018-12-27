import heapq
from collections import namedtuple

import cv2
import numpy as np
import networkx as nx

from PanoramaStitching import Utils


def get_inverted_difference_image(image_a, image_b):
    diff = image_a - image_b
    # max = np.amax(diff, axis=2)
    return 255 - diff


def get_3x3_neighbour_elements(matrix, x, y):
    width, height = matrix.shape[:2]
    neighbour = np.ones((3, 3), dtype=int)
    neighbour = np.negative(neighbour)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i != 0 or j != 0) and (width >= x + i >= 0) and (height >= y + j >= 0):
                neighbour[i + 1, j + 1] = matrix[x + i, y + j]
    return neighbour


def get_watershed_image(image, mask):
    height, width = mask.shape
    original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    original_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    ext_left = tuple(contours[0][contours[0][:, :, 0].argmin()][0])[0]
    ext_right = tuple(contours[0][contours[0][:, :, 0].argmax()][0])[0]

    gray = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = np.where(mask == 0, mask, gray)
    unknown = cv2.subtract(mask, gray)
    # cv2.imshow('gray', gray)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=2)
    half = int(ext_left + ((ext_right - ext_left) / 2))
    for i in range(height):
        if mask[i, half] != 255:
            gray[i, half] = 255
    cv2.imshow('gray2', gray)
    cv2.waitKey()

    ret, markers = cv2.connectedComponents(gray)
    # markers = markers + 1
    # markers[unknown == 255] = 0
    label_hue = np.uint8(179 * markers / np.max(markers))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # cv2.imshow('labeled.png', labeled_img)
    # cv2.waitKey()

    watershed_image = cv2.watershed(original_image, markers)
    original_mask[watershed_image == -1] = [0, 0, 255]
    """
    stepped_over = np.zeros(height, dtype=bool)
    for y in range(width):
        for x in range(height):
            if mask[x, y] == 255:
                if (not stepped_over[x]) and watershed_image[x, y] == 9:
                    watershed_image[x, y] = -2
                elif watershed_image[x, y] == -1:
                    stepped_over[x] = True

    new_watershed = watershed_image
    iteration = 0
    # cv2.waitKey()
    while True:
        iteration += 1
        for (x, y), item in np.ndenumerate(watershed_image):
            if (y == 180) and x == 121:
                print()
            if item == 9 and mask[x, y] == 255:
                neighbours = get_3x3_neighbour_elements(watershed_image, x, y)
                if np.any(neighbours == -2):
                    new_watershed[x, y] = -2
        if (new_watershed == watershed_image).all():
            break
        else:
            watershed_image = new_watershed

    temp_labels = np.unique(watershed_image[1, :])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = temp_labels[1]

    temp_labels = np.unique(watershed_image[height - 2, :])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = temp_labels[1]
    """
    temp_labels = np.unique(watershed_image[:, 1])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = -5

    temp_labels = np.unique(watershed_image[:, width - 2])
    for item in temp_labels:
        if item != -1:
            watershed_image[watershed_image == item] = -10


    # original_mask[watershed_image == -2] = [255, 0, 0]
    cv2.imshow('water', original_mask)
    cv2.waitKey()

    return watershed_image


def find_cut_seam(segmented_image, inverted_image, mask):
    original_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    height, width = segmented_image.shape
    Edge = namedtuple("Edge", ["node_1", "node_2"])
    edges = {}
    graph = nx.Graph()
    for x in range(height):
        for y in range(width):
            if mask[x, y] == 255 and segmented_image[x, y] == -1:
                """
                cv2.circle(original_mask, (y, x), 5, (0, 255, 0))
                cv2.imshow('point', original_mask)
                cv2.waitKey()
                """
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
    print(cut_nodes)
    """
    segments = []
    for cut in cut_nodes:
        if cut[0] > 1:
            segments.append(cut[0])
        elif
    for x in width:
        for y in height:
            if
    """
    return cut_nodes


def graph_cut_blend(image_a, image_b):
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
    new_image_b = image_b
    copy_black = False
    current_line = 0
    for (x, y), item in np.ndenumerate(segmented_image):
        if current_line != x:
            copy_black = False
        if x == 0 or y == 0 or x == height - 1 or y == width - 1:
            continue
        if x == 192:
            print()
        if segmented_image[x, y] == -1 \
            and segmented_image[x, y - 1] != segmented_image[x, y + 1] \
                and (np.any((segmented_image[x, y - 1], segmented_image[x, y + 1]) == cut_nodes)
                     or np.any((segmented_image[x, y + 1], segmented_image[x, y - 1]) == cut_nodes)):
            copy_black = True
            current_line = x
        if copy_black:
            new_image_b[x, y] = [0, 0, 0]

    cv2.imshow('new_im', new_image_b)
    cv2.waitKey()
