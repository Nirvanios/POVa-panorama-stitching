

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
