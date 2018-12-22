from enum import Enum
import cv2
import numpy as np


class KeyPointDetector(Enum):
    SIFT = 1
    SURF = 2


class Matcher:

    def __init__(self, detector_type, matches_required, enable_log=True):
        if detector_type == KeyPointDetector.SIFT:
            self.featureDetector = cv2.xfeatures2d.SIFT_create()
        if detector_type == KeyPointDetector.SURF:
            self.featureDetector = cv2.xfeatures2d.SURF_create()

        self.matcher = cv2.DescriptorMatcher_create("BruteForce")

        self.enable_log = enable_log

        self.matches_required = matches_required

    def detect_and_describe(self, image):
        """
        Get key points detected by SIFT/SURF
        :param image: input image
        """
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.featureDetector.detectAndCompute(gray_scale_image, None)

        key_points = np.float32([key_point.pt for key_point in key_points])
        return key_points, descriptors

    def match_key_points(self, key_points_a, key_points_b, descriptors_a, descriptors_b,
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
        raw_matches = self.matcher.knnMatch(descriptors_a, descriptors_b, 2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if self.enable_log:
            print("[DEBUG] matches: " + str(len(matches)))
        if len(matches) > self.matches_required:
            points_a = np.float32([key_points_a[i] for (_, i) in matches])
            points_b = np.float32([key_points_b[i] for (i, _) in matches])

            (H, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC,
                                             threshold)

            return matches, H, status

        return None

    def show_matches(self, image_a, image_b, key_points_a, key_points_b, matches, status):
        """
        Show window with matched keypoints
        :param image_a: first image
        :param image_b: second image
        :param key_points_a: key points of the first image
        :param key_points_b: key points of the second image
        :param matches:
        :param status:
        """
        (hA, wA) = image_a.shape[:2]
        (hB, wB) = image_b.shape[:2]
        result = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
        result[0:hA, 0:wA] = image_a
        result[0:hB, wA:] = image_b

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                point_a = (int(key_points_a[queryIdx][0]), int(key_points_a[queryIdx][1]))
                point_b = (int(key_points_b[trainIdx][0]) + wA, int(key_points_b[trainIdx][1]))
                cv2.line(result, point_a, point_b, (0, 255, 0), 1)

        cv2.imshow('matches', result)
        cv2.waitKey()
