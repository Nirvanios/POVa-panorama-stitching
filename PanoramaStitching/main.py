import argparse
import datetime
import linecache
import os
import tracemalloc

import cv2
import numpy as np

import PanoramaStitching.Utils as PanoUtils
from PanoramaStitching.Matcher import KeyPointDetector
from PanoramaStitching.Matcher import Matcher
from PanoramaStitching.PanoramaImage import MainPanoramaImage
import PanoramaStitching.Stitcher as Stitcher
from PanoramaStitching.Logger import logger_instance
from PanoramaStitching.Logger import LogLevel

show_mem = False

parser = argparse.ArgumentParser(description='Panorama stitching.')
parser.add_argument('--folder', help='path to file with images')
parser.add_argument('--img', help='path to main image')
parser.add_argument('--dest', help='destination of output image')
parser.add_argument('--out', help='destination of debug output')
parser.add_argument('--sift', default=True, help='Use SIFT for key point detection')
parser.add_argument('--surf', default=False, help='Use SURF for key point detection')


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

def panorama(args, main_image, images):
    logger_instance.log(LogLevel.STATUS, "Usign homography")
    logger_instance.log(LogLevel.STATUS, "Calculating image descriptors")
    for img in images:
        img.calculate_descriptors(matcher)

    panorama_image = MainPanoramaImage(main_image.name, main_image.image)
    main_image.checked = True

    added = True
    cnt = 0
    logger_instance.log(LogLevel.STATUS, "Starting main panorama loop")
    while added:
        added = False

        panorama_image.calculate_descriptors(matcher)

        panorama_image.calculate_matches(images, matcher)

        match_count, index = panorama_image.find_best_match()

        logger_instance.log(LogLevel.DEBUG, "Index: " + str(index) + " Cnt: " + str(match_count))

        if not index == -1:
            matches, h, status = panorama_image.matches[index][0]
            panorama_image.image = Stitcher.stitch_images_homography(panorama_image.image,
                                                                     panorama_image.matches[index][1].image,
                                                                     h)
            panorama_image.image = PanoUtils.crop_black(panorama_image.image)
            logger_instance.log(LogLevel.INFO, "Stitching with: " + panorama_image.matches[index][1].name)
            panorama_image.matches[index][1].checked = True
            added = True
            save_location = args.out + str(cnt) + ".png"
            logger_instance.log(LogLevel.DEBUG, "saving intermediate result to: " + save_location)
            cv2.imwrite(save_location, panorama_image.image)
            cnt += 1
            logger_instance.log(LogLevel.STATUS, "Matched " + str(cnt + 1) + "/" + str(len(images)) + " images")

    w, h, _ = panorama_image.image.shape

    cv2.imshow('pano', panorama_image.image)
    logger_instance.log(LogLevel.STATUS, "Saving finished panorama image")
    cv2.imwrite(args.dest, panorama_image.image)
    cv2.waitKey()

    logger_instance.log(LogLevel.INFO, "List of images used in panorama:")
    for img in images:
        if img.checked:
            logger_instance.log(LogLevel.NONE, "\t\t" + img.name)

    logger_instance.log(LogLevel.INFO, "List of images NOT used in panorama:")
    for img in images:
        if not img.checked:
            logger_instance.log(LogLevel.NONE, "\t\t" + img.name)


def panorama_affine(args, main_image, images):
    logger_instance.log(LogLevel.STATUS, "Using affine")
    f = 400

    logger_instance.log(LogLevel.STATUS, "Calculating image descriptors")
    for img in images:
        h, w = img.image.shape[:2]
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        img.image = Stitcher.cylindricalWarpImage(img.image, K)[0]
        img.image = cv2.copyMakeBorder(img.image, 500, 500, 500, 500, cv2.BORDER_CONSTANT)
        img.calculate_descriptors(matcher)

    panorama_image = MainPanoramaImage(main_image.name, main_image.image)
    main_image.checked = True

    added = True
    cnt = 0
    logger_instance.log(LogLevel.STATUS, "Starting main panorama loop")
    while added:
        added = False

        panorama_image.calculate_descriptors(matcher)

        panorama_image.calculate_matches(images, matcher, "affine")

        match_count, index = panorama_image.find_best_match()

        logger_instance.log(LogLevel.DEBUG, "Index: " + str(index) + " Cnt: " + str(match_count))

        if not index == -1:
            matches, h, status = panorama_image.matches[index][0]
            panorama_image.image = Stitcher.stitch_images_affine(panorama_image.image,
                                                                 panorama_image.matches[index][1].image,
                                                                 h)
            # panorama_image.image = PanoUtils.crop_black(panorama_image.image)
            logger_instance.log(LogLevel.INFO, "Stitching with: " + panorama_image.matches[index][1].name)
            panorama_image.matches[index][1].checked = True
            added = True
            save_location = args.out + str(cnt) + ".png"
            logger_instance.log(LogLevel.DEBUG, "saving intermediate result to: " + save_location)
            cv2.imwrite(save_location, panorama_image.image)
            cnt += 1
            logger_instance.log(LogLevel.STATUS, "Matched " + str(cnt + 1) + "/" + str(len(images)) + " images")


def main(args):
    images = PanoUtils.load_images(args.folder)
    logger_instance.log(LogLevel.STATUS, "Images in folder:")
    for img in images:
        logger_instance.log(LogLevel.NONE, "\t\t" + img.name)

    for x in images:
        if x.name == os.path.basename(args.img):
            break
    else:
        x = None

    if x is None:
        logger_instance.log(LogLevel.ERROR, "File \"" + os.path.basename(args.img) + "\" not found")
        return

    main_image = x
    logger_instance.log(LogLevel.STATUS, "Main image:")
    logger_instance.log(LogLevel.NONE, "\t\t" + main_image.name)

    global matcher
    if args.sift:
        logger_instance.log(LogLevel.STATUS, "Usign SIFT")
        matcher = Matcher(KeyPointDetector.SIFT, 4)
    if args.surf:
        logger_instance.log(LogLevel.STATUS, "Usign SURF")
        matcher = Matcher(KeyPointDetector.SURF, 4)

    panorama(args, main_image, images)


if __name__ == "__main__":
    if show_mem:
        tracemalloc.start()

    start_time = datetime.datetime.now()
    try:
        main(parser.parse_args())
    except Exception as e:
        logger_instance.log_exc(e)
    end_time = datetime.datetime.now()
    logger_instance.log(LogLevel.INFO, "Total execution time: " + str((end_time - start_time)))
