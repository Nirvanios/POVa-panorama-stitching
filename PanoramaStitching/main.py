import argparse

import cv2

import tracemalloc
from collections import Counter
import linecache
import os

import PanoramaStitching.Utils as PanoUtils
from PanoramaStitching.Matcher import KeyPointDetector
from PanoramaStitching.Matcher import Matcher
from PanoramaStitching.Stitcher import Stitcher

show_mem = False

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')
parser.add_argument('--sift', default=True)
parser.add_argument('--surf', default=False)


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main(args):
    global matcher
    if args.sift:
        matcher = Matcher(KeyPointDetector.SIFT, 100)
    if args.surf:
        matcher = Matcher(KeyPointDetector.SURF, 100)
    images = PanoUtils.load_images(args.folder)

    for img in images:
        img.calculate_descriptors(matcher)

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
            kp_a, desc_a = panorama.get_descriptors()
            kp_b, desc_b = images[i].get_descriptors()
            m = matcher.match_key_points(kp_a, kp_b, desc_a, desc_b, 0.75, 4.5)
            if m is None:
                print("[INFO] not enough matches, skipping")
                print("__________________________________________________________________")
                continue
            added = True
            images[i].checked = True
            matches, H, status = m

            print("[INFO] matched, stitching")
            panorama.image = Stitcher.stitch_images(images[i].image, panorama.image, H)
            panorama.image = PanoUtils.crop_black(panorama.image)

            save_location = "/Users/petr/Desktop/images/output/out" + str(cnt) + ".png"
            print("[DEBUG] saving intermediate result to: " + save_location)
            cv2.imwrite(save_location, panorama.image)
            cnt += 1
            if show_mem:
                snapshot = tracemalloc.take_snapshot()
                display_top(snapshot)
            print("__________________________________________________________________")
        print("[INFO] next iteration")
    print("[INFO] stitching done, matched " + str(cnt) + "/" + str(len(images) - 1))

    cv2.imwrite(args.dest, panorama.image)
    cv2.waitKey()


if __name__ == "__main__":
    if show_mem:
        tracemalloc.start()
    main(parser.parse_args())
