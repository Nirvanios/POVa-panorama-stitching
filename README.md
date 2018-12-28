# POVa-panorama-stitching

Project for Computer Vision course on VUT FIT.

Multiple image panorama stitching with blending.


usage: main.py [-h] --folder FOLDER --img IMG --dest DEST [--out OUT]
               [--kp_detector {SIFT,SURF}] [--pano_type {HOMOGRAPHY,AFFINE}]
               [--cyl_wrap] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       path to file with images
  --img IMG             path to main image
  --dest DEST           destination of output image
  --out OUT             destination of debug output
  --kp_detector {SIFT,SURF}
                        key point detector SIFT or SURF
  --pano_type {HOMOGRAPHY,AFFINE}
                        type of panorama HOMOGRAPHY or AFFINE
  --cyl_wrap            Wrap images on a cylinder
  --debug               Allow debug prints
