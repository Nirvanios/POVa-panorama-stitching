import string
import sys
import cv2
import numpy as np
import argparse
import glob
import os

from src.Matcher import Matcher
from src.Matcher import KeyPointDetector
import src.Utils as PanoUtils

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')
parser.add_argument('--sift', default=True)
parser.add_argument('--surf', default=False)





def stitch_images(image_a, image_b, hom):
    """
    Warp image_b by given matrix (hom) and stitch it together with image_a
    :param image_a: first image
    :param image_b: second image
    :param hom: homography matrix
    :return: stitched image
    """
    result = cv2.warpPerspective(image_a, hom, (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
    return result


def crop_black(image):
    """
    Crop black parts of the image
    :param image: input image
    :return: cropped image
    """
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = image[y:y + h, x:x + w]
    return crop


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img


def main(args):
    global matcher
    if args.sift:
        matcher = Matcher(KeyPointDetector.SIFT, 100)
    if args.surf:
        matcher = Matcher(KeyPointDetector.SURF, 100)
    images = PanoUtils.load_images(args.folder)

    panorama = images[8]
    print("[INFO] Base image: " + images[8].name)
    added = True
    cnt = 0
    while added:
        added = False
        for i in range(len(images)):
            if images[i].checked:
                continue
            print("[INFO] Current image: " + images[i].name)
            panorama.calculate_descriptors(matcher)
            images[i].calculate_descriptors(matcher)
            kp_a, desc_a = panorama.get_descriptors()
            kp_b, desc_b = images[i].get_descriptors()
            m = matcher.match_key_points(kp_a, kp_b, desc_a, desc_b, 0.75, 4.5)
            if m is None:
                continue
            added = True
            images[i].checked = True
            matches, H, status = m

            panorama.image = warpImages(images[i].image, panorama.image, H)
            panorama.image = crop_black(panorama.image)
            cv2.imwrite("/Users/petr/Desktop/images/output/out" + str(cnt) + ".png", panorama.image)
            cnt += 1
            print("[INFO] " + str(i) + "/" + str(len(images) - 1))
        print("[INFO] next iteration")
    print("[INFO] stitching done, matched " + str(cnt) + "/" + str(len(images) - 1))

    cv2.imwrite(args.dest, panorama)
    cv2.waitKey()


if __name__ == "__main__":
    main(parser.parse_args())
