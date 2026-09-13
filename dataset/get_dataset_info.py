import os
from PIL import Image


IMAGES_DIR = 'images/'
files = [f'{IMAGES_DIR}{filename}' for filename in os.listdir(IMAGES_DIR)]

resolutions = set()
for file in files:
    resolutions.add(Image.open(file).size)

print(resolutions)
