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
    channels = cv2.split(image_a)
    max_a = cv2.max(channels[0], cv2.max(channels[1], channels[2]))
    channels = cv2.split(image_b)
    max_b = cv2.max(channels[0], cv2.max(channels[1], channels[2]))
    diff = max_a - max_b
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
    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    # gray = cv2.dilate(gray, kernel, iterations=2)

    gray[mask != 255] = 255
    """
    cv2.imshow("diff", gray)
    cv2.waitKey()
    """
    half = int(ext_left + ((ext_right - ext_left) / 2))
    for i in range(height):
        if gray[i, half] == 255:
            gray[i, half] = 0
        else:
            break

    for i in range(height - 1, -1, -1):
        if gray[i, half] == 255:
            gray[i, half] = 0
        else:
            break
    """
    cv2.imshow("diff", gray)
    cv2.waitKey()
    """

    ret, markers = cv2.connectedComponents(gray)

    watershed_image = cv2.watershed(original_image, markers)
    original_mask[watershed_image == -1] = [0, 0, 255]
    """
    cv2.imshow("diff", original_mask)
    cv2.waitKey()
    """

    original_mask[watershed_image == 1] = [255, 0, 0]
    original_mask[watershed_image == 2] = [0, 255, 0]
    original_mask[watershed_image == -6] = [255, 0, 255]
    original_mask[watershed_image == -1] = [0, 0, 255]
    """
    cv2.imshow("diff", original_mask)
    cv2.waitKey()
    """

    return watershed_image


def find_cut_seam(segmented_image, inverted_image, mask):
    """
    Creates graph and perform minimal cut on it.
    :param segmented_image: segmented image from get_watershed_image function
    :param inverted_image: inverted difference of two images
    :param mask: mask of overlapping region
    :return: List of node pairs where is minimal cut found
    """
    height, width = segmented_image.shape
    Edge = namedtuple("Edge", ["node_1", "node_2"])
    edges = {}
    graph = nx.Graph()
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            if mask[x, y] == 255 and segmented_image[x, y] == -1:

                edge = None
                edge_inv = None

                up = segmented_image[x - 1, y]
                down = segmented_image[x + 1, y]
                left = segmented_image[x, y - 1]
                right = segmented_image[x, y + 1]
                if up != down and up != -1 and down != -1:
                    edge = Edge(node_1=up, node_2=down)
                    edge_inv = Edge(node_1=down, node_2=up)
                elif left != right and left != -1 and right != -1:
                    edge = Edge(node_1=left, node_2=right)
                    edge_inv = Edge(node_1=right, node_2=left)

                if edge is None:
                    continue

                weight = 1
                if edge.node_1 == 1 or edge.node_2 == 2 or edge_inv.node_1 == 1 or edge_inv.node_2 == 2:
                    weight = 2

                if edge in edges:
                    edges[edge] += inverted_image[x, y] * weight
                elif edge_inv in edges:
                    edges[edge_inv] += inverted_image[x, y] * weight
                else:
                    edges[edge] = inverted_image[x, y].astype(int) * weight

    for edge in edges:
        graph.add_edge(edge.node_1, edge.node_2, cardinality=edges[edge])
    cut_nodes = nx.algorithms.connectivity.minimum_edge_cut(graph, 1, 2)

    pass

    return cut_nodes, graph


def graph_cut_blend(image_a, mask_a, image_b, mask_b, center_a, center_b):
    """
    Blends two images using watershed and graph cut.
    :param image_a: First image
    :param image_b: Second image
    :return: Panorama blended using graph cut
    """

    Utils.cut_pixels_around_image(image_a, mask_a)
    Utils.cut_pixels_around_image(image_b, mask_b)

    """
    cv2.imshow("imageA", image_a)
    cv2.imshow("maskA", mask_a)
    cv2.imshow("maskB", mask_b)
    cv2.waitKey()
    """

    height, width = image_a.shape[:2]
    #gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    #gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    difference_image = get_inverted_difference_image(image_a, image_b)
    mask = Utils.get_overlapping_mask(mask_a, mask_b)
    difference_image = np.where(mask == 255, difference_image, 0)

    segmented_image = get_watershed_image(difference_image, mask)

    cut_nodes, graph = find_cut_seam(segmented_image, difference_image, mask)
    cut_nodes = np.array(list(cut_nodes))
    print(cut_nodes)
    seam_image = highlight_seam(segmented_image, cut_nodes, mask)
    segmented_mask = highlight_segments(cut_nodes, graph, segmented_image)

    new_imageA = np.copy(image_a)

    if center_a[0][1] > center_b[0][1]:
        segmented_mask = cv2.bitwise_not(segmented_mask)


    locs = np.where(mask_b == 255)
    new_imageA[locs[0], locs[1]] = image_b[locs[0], locs[1]]



    new_mask = cv2.bitwise_and(mask, segmented_mask)

    locs = np.where(new_mask == 255)
    new_imageA[locs[0], locs[1]] = image_a[locs[0], locs[1]]
    """
    cv2.imshow("imageA", image_a)
    cv2.imshow("imageB", image_b)
    cv2.waitKey()
    """

    """
    cv2.imshow("new_imageA", new_imageA)
    cv2.waitKey()
    """

    return new_imageA


def highlight_seam(input_image, cuts, mask):
    bck = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    black = np.zeros(mask.shape, dtype=np.uint8)
    height, width = input_image.shape
    tmp_image = np.copy(input_image)
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            if mask[x, y] == 255 and input_image[x, y] == -1:
                bck[x, y] = (255, 0, 0)
                up = input_image[x - 1, y]
                down = input_image[x + 1, y]
                left = input_image[x, y - 1]
                right = input_image[x, y + 1]
                if up != down and up != -1 and down != -1:
                    if (up in cuts[:, 0] and down in cuts[:, 1]) or (down in cuts[:, 0] and up in cuts[:, 1]):
                        bck[x, y] = (0, 0, 255)
                        black[x, y] = 255
                        tmp_image[x, y] = -2
                elif left != right and left != -1 and right != -1:
                    if (left in cuts[:, 0] and right in cuts[:, 1]) or (right in cuts[:, 0] and left in cuts[:, 1]):
                        bck[x, y] = (0, 0, 255)
                        black[x, y] = 255
                        tmp_image[x, y] = -2

    input_image = tmp_image

    """
    cv2.imshow("bck", bck)
    cv2.waitKey()
    """

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)
    """
    cv2.imshow("bck", closing)
    cv2.waitKey()
    """
    return closing


def highlight_segments(cuts, graph, input_image):
    mask = np.zeros(input_image.shape, dtype=np.uint8)
    graph.remove_edges_from(cuts)
    connected_labels = set()
    labels_to_walk = {1}
    while len(labels_to_walk) is not 0:
        current_label = labels_to_walk.pop()
        connected_labels.add(current_label)
        new_labels = set([e for e in graph[current_label]])
        new_labels = new_labels - connected_labels
        labels_to_walk.update(new_labels)

    for label in connected_labels:
        mask[input_image == label] = 255

    """
    cv2.imshow("mask", mask)
    cv2.waitKey()
    """

    ret, markers = cv2.connectedComponents(mask, connectivity=8)
    markers = markers.astype(np.uint8)
    mask2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY)[1]
    """
    cv2.imshow("bck", mask2)
    cv2.waitKey()
    """
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    """
    cv2.imshow("bck", closing)
    cv2.waitKey()
    """

    return closing


