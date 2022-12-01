import os
import cv2
from glob import glob

imgs = sorted([img for img in glob('/data1/cenchaojun/tracer/data/PASCAL-S/Test/images/*.png')])
masks = sorted([mask for mask in glob('/data1/cenchaojun/tracer/data/PASCAL-S/Test/masks/*.png')])

for image_path, mask_path in zip(imgs,masks):
    img = cv2.imread(image_path)
    mask =cv2.imread(mask_path)
    img_w = img.shape[1]
    img_h =img.shape[0]

    mask_w = mask.shape[1]
    mask_h =mask.shape[0]

    if img_w!= mask_w or img_h!=mask_h:
        print(image_path)