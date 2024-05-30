import os
import sys
import numpy as np
from scipy import misc, ndimage
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("int_class", type=int, help="category of class in integer type")
parser.add_argument("class_name", type=str, help="category of class that images are belonged to")
args = parser.parse_args()
# Load the original image

prepath = os.path.join(os.getcwd(), 'raw', 'trash_images_test')
class_dir = os.path.join(prepath, str(args.int_class))
destPath = os.path.join(os.getcwd(), 'processed', 'trash_images_test_resized', args.class_name)

print(prepath)