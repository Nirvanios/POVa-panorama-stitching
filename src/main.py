import argparse

import cv2

import src.Utils as PanoUtils
from src.Matcher import KeyPointDetector
from src.Matcher import Matcher
from src.Stitcher import Stitcher

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')
parser.add_argument('--sift', default=True)
parser.add_argument('--surf', default=False)


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

            panorama.image = Stitcher.stitch_images(images[i].image, panorama.image, H)
            panorama.image = PanoUtils.crop_black(panorama.image)
            cv2.imwrite("/Users/petr/Desktop/images/output/out" + str(cnt) + ".png", panorama.image)
            cnt += 1
            print("[INFO] " + str(i) + "/" + str(len(images) - 1))
        print("[INFO] next iteration")
    print("[INFO] stitching done, matched " + str(cnt) + "/" + str(len(images) - 1))

    cv2.imwrite(args.dest, panorama)
    cv2.waitKey()


if __name__ == "__main__":
    main(parser.parse_args())
