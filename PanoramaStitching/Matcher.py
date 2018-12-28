from enum import Enum
import cv2
import numpy as np

from PanoramaStitching.Logger import logger_instance, LogLevel


# Set key point detector enum
class KeyPointDetector(Enum):
    SIFT = 1
    SURF = 2


class Matcher:

    def __init__(self, detector_type, matches_required):
        if detector_type == KeyPointDetector.SIFT:
            self.key_point_detector = cv2.xfeatures2d.SIFT_create()
        if detector_type == KeyPointDetector.SURF:
            self.key_point_detector = cv2.xfeatures2d.SURF_create()

        # setup FLANN detector
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.desc_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.matches_required = matches_required

    def detect_and_describe(self, image):
        """
        Get key points detected by SIFT/SURF
        :param image: input image
        """

        # Detect in gray scaled image
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.key_point_detector.detectAndCompute(gray_scale_image, None)

        # Convert
        key_points = np.float32([key_point.pt for key_point in key_points])
        return key_points, descriptors

    def match_key_points(self, key_points_a, key_points_b, descriptors_a, descriptors_b,
                         ratio, threshold, method="homography"):
        """
        match key points given by SIFT/SURF
        :param method: homography/affine
        :param key_points_a: key points of first image
        :param key_points_b: key points of second image
        :param descriptors_a: descriptors of first image
        :param descriptors_b: descriptors of second image
        :param ratio:
        :param threshold:
        :return: matches and homography/affine matrix
        """

        # Find matches
        raw_matches = self.desc_matcher.knnMatch(descriptors_a, descriptors_b, 2)
        matches = []

        # Save matches
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        logger_instance.log(LogLevel.DEBUG, "matches: " + str(len(matches)))

        # Estimate geometrical transformation
        if len(matches) > self.matches_required:
            points_a = np.float32([key_points_a[i] for (_, i) in matches])
            points_b = np.float32([key_points_b[i] for (i, _) in matches])

            if method == "homography":
                (H, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC,
                                                 threshold)
            elif method == "affine":
                (H, status) = cv2.estimateAffine2D(points_a, points_b, cv2.RANSAC,
                                                   ransacReprojThreshold=threshold)
            else:
                raise Exception("Invalid call, unsupported transformation type: " + method)

            return matches, H, status

        return None

    @staticmethod
    def show_matches(image_a, image_b, key_points_a, key_points_b, matches, status):
        """
        Show window with matched key points
        :param image_a: first image
        :param image_b: second image
        :param key_points_a: key points of the first image
        :param key_points_b: key points of the second image
        :param matches:
        :param status:
        """

        # Create view
        h_a, w_a = image_a.shape[:2]
        h_b, w_b = image_b.shape[:2]
        result = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype="uint8")
        result[0:h_a, 0:w_a] = image_a
        result[0:h_b, w_a:] = image_b

        # Add line between detected points
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                point_a = (int(key_points_a[queryIdx][0]), int(key_points_a[queryIdx][1]))
                point_b = (int(key_points_b[trainIdx][0]) + w_a, int(key_points_b[trainIdx][1]))
                cv2.line(result, point_a, point_b, (0, 255, 0), 1)

        cv2.imshow('matches', result)
        cv2.waitKey()
