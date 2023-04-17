import numpy as np
import cv2
import os
from PIL import Image
path = "./img_datas/test/label"
max_label = 0
imgs = os.listdir(path)
print('imgs:', imgs)
for i in imgs:
    img_dir = os.path.join(path, i)
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # img[img > 0] = 1
    # new_path = './img_datas/train/newlabel/' + i
    # print('i:', new_path)
    # cv2.imwrite(new_path, img)
    temp = np.max(img)
    # print('i:', temp)
    if temp > max_label:
        max_label = temp
print('max_label:', max_label)