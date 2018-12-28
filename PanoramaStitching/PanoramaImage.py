

class PanoramaImage:
    checked = False
    key_points = None
    descriptors = None

    def __init__(self, name, image):
        self.name = name
        self.image = image

    def calculate_descriptors(self, matcher):
        """
        Calculate descriptors in image matches
        :param matcher: matcher of images
        """
        self.key_points, self.descriptors = matcher.detect_and_describe(self.image)

    def get_descriptors(self):
        """
        Getter to print key points and descriptors
        :return key_points, descriptors
        """
        return self.key_points, self.descriptors


class MainPanoramaImage(PanoramaImage):
    matches = []

    def calculate_matches(self, images, matcher, method="homography"):
        """
        Calculate matches between panorama image and remaining unused images
        :param method: homography/affine
        :param images: list of images
        :param matcher: key point matcher
        """
        self.matches.clear()

        for img in images:
            if not img.checked:
                # Try to match key points
                m = matcher.match_key_points(img.key_points, self.key_points,
                                             img.descriptors, self.descriptors, 0.7,
                                             4.5, method)
                if m is None:
                    continue

                self.matches.append((m, img))

    def find_best_match(self):
        """
        Find best match for stitching.
        :return: amount of matches and index of the image
        """
        max_matches = 0
        max_index = -1
        cnt = 0

        # Iterate over matches and save count
        for match in self.matches:
            match_count = len(match[0][0])
            if match_count > max_matches:
                max_matches = match_count
                max_index = cnt
            cnt += 1

        return max_matches, max_index
