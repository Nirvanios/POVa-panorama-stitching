import string
import sys
import cv2
import numpy as np
import argparse
import glob
import os

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create("BruteForce")

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')


def load_images(folder):
    """
    Get list of images in a folder
    :param folder: folder containing images
    :return: list of images and their names
    """
    filenames = glob.glob(folder + "/*.png")
    images = [cv2.imread(img) for img in filenames]
    ret = []
    for i in range(len(filenames)):
        ret.append([images[i], os.path.basename(filenames[i])])
    return ret


def get_sift(image):
    """
    Get key points detected by SIFT
    :param image: input image
    """
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray_scale_image, None)

    key_points = np.float32([key_point.pt for key_point in key_points])
    return key_points, descriptors


def match_keypoints(key_points_a, key_points_b, descriptors_a, descriptors_b,
                    ratio, threshold):
    """
    match key points given by SIFT
    :param key_points_a: key points of first image
    :param key_points_b: key points of second image
    :param descriptors_a: descriptors of first image
    :param descriptors_b: descriptors of second image
    :param ratio:
    :param threshold:
    :return: matches and homography if
    """
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, 2)
    matches = []

    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        points_a = np.float32([key_points_a[i] for (_, i) in matches])
        points_b = np.float32([key_points_b[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC,
                                         threshold)

        return matches, H, status

    return None


def show_matches(image_a, image_b, key_points_a, key_points_b, matches, status):
    (hA, wA) = image_a.shape[:2]
    (hB, wB) = image_b.shape[:2]
    result = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    result[0:hA, 0:wA] = image_a
    result[0:hB, wA:] = image_b

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            point_a = (int(key_points_a[queryIdx][0]), int(key_points_a[queryIdx][1]))
            point_b = (int(key_points_b[trainIdx][0]) + wA, int(key_points_b[trainIdx][1]))
            cv2.line(result, point_a, point_b, (0, 255, 0), 1)

    cv2.imshow('matches', result)
    cv2.waitKey()

def stitch_images(image_a, image_b, hom):
    result = cv2.warpPerspective(image_a, hom, (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
    return result


def main(args):
    images = load_images(args.folder)
    image_a = images[0][0]
    image_b = images[1][0]
    kp_a, desc_a = get_sift(image_a)
    kp_b, desc_b = get_sift(image_b)

    matches, H, status = match_keypoints(kp_a, kp_b, desc_a, desc_b, 0.75, 4.5)

    #show_matches(images[3][0], images[1][0], kp_a, kp_b, matches, status)
    cv2.imshow('stitch result', stitch_images(image_a, image_b, H))
    cv2.waitKey()


if __name__ == "__main__":
    main(parser.parse_args())
