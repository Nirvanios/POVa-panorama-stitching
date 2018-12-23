from PanoramaStitching.Matcher import Matcher

class PanoramaImage:
    checked = False
    key_points = None
    descriptors = None

    def __init__(self, name, image):
        self.name = name
        self.image = image

    def calculate_descriptors(self, matcher):
        self.key_points, self.descriptors = matcher.detect_and_describe(self.image)

    def get_descriptors(self):
        return self.key_points, self.descriptors


class MainPanoramaImage(PanoramaImage):
    matches = []

    def calculate_matches(self, images, matcher):
        self.matches.clear()

        for img in images:
            if not img.checked:
                m = matcher.match_key_points(self.key_points, img.key_points, self.descriptors, img.descriptors, 0.7, 4.5)
                if m is None:
                    continue

                self.matches.append((m, img))

    def find_best_match(self):
        max_matches = 0
        max_index = -1
        cnt = 0
        for match in self.matches:
            match_count = len(match[0][0])
            if match_count > max_matches:
                max_matches = match_count
                max_index = cnt
            cnt += 1

        return max_matches, max_index
