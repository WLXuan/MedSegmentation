import glob
import cv2
import numpy as np
import os


def npz(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(images_path)
    for s in images:
        image_path = os.path.join(images_path, s)
        label_path = os.path.join(labels_path, s)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 标签由三通道转换为单通道
        label = cv2.imread(label_path, flags=0)
        # 保存npz文件
        np.savez(path2 + s[:-4] + ".npz", image=image, label=label)


npz('./img_datas/train/image/', './img_datas/train/label/', './data/Synapse/train_npz/train_npz')

npz('./img_datas/test/image/', './img_datas/test/label/', './data/Synapse/test_vol_h5/test_vol_h5')
