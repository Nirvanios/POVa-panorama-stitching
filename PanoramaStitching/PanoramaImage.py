import threading

from PanoramaStitching.Logger import logger_instance, LogLevel


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

    def match(self, start, end, images, matcher):
        for i in range(start, end):
            if not images[i].checked:
                m = matcher.match_key_points(images[i].key_points, self.key_points,
                                             images[i].descriptors, self.descriptors, 0.7,
                                             4.5)
                if m is None:
                    continue

                self.matches.append((m, images[i]))

    def calculate_matches(self, images, matcher):
        """
        Calculate matches between panorama image and remaining unused images
        :param images: list of images
        :param matcher: key point matcher
        :return:
        """
        self.matches.clear()

        #logger_instance.log(LogLevel.DEBUG, "Starting thread for match calc")
        #t1 = threading.Thread(target=self.match, args=(0, int(len(images) / 2), images, matcher))
        #t1.start()
        #self.match(int(len(images) / 2), len(images), images, matcher)
        #t1.join()
        #logger_instance.log(LogLevel.DEBUG, "Match calc done")
        for img in images:
            if not img.checked:
                m = matcher.match_key_points(img.key_points, self.key_points,
                                             img.descriptors, self.descriptors, 0.7,
                                             4.5)
                if m is None:
                    continue

                self.matches.append((m, img))

    def find_best_match(self):
        """
        Find best match for stitching
        :return: amount of matches and index of the image
        """
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
