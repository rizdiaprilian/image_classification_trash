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

prepath = os.path.join(os.getcwd(), 'raw', 'trash_images_train')
class_dir = os.path.join(prepath, str(args.int_class))
destPath = os.path.join(os.getcwd(), 'processed', 'trash_images_train_resized', args.class_name)

print(os.getcwd())

def resize(image, dim1, dim2):
	return misc.imresize(image, (dim1, dim2))

try: 
	os.makedirs(destPath)
except OSError:
	if not os.path.isdir(destPath):
		raise

for subdir, dirs, files in os.walk(class_dir):
    for file in files:
        if len(file) <= 4 or file[-4:] != '.jpg':
            print(file)

        img = Image.open(os.path.join(subdir, file))

        # Resize the image
        new_width, new_height = 384, 512
        resized_img = img.resize((new_width, new_height))
        resized_img.save(os.path.join(destPath, file))
		
print(destPath)
    