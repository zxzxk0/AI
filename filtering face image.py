#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import cv2

import numpy as np


TARGET_PIXEL = 128
DOWNSTREAM = 8
SAVE_PATH = r"/data01/notebooks/jh/skin"


def make_directory(path):
    os.makedirs(path, exist_ok=True)


def get_image_paths(base):
    ret = []
    for path, dirs, files in os.walk(base):
        for file in files:
            if file.endswith("labels.png"):
                ret.append((os.path.join(path, file.replace("_labels.png", ".jpg")), os.path.join(path, file)))
    return ret


def execute_patching(img_path, mask_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    height, width, _ = image.shape
    fname = os.path.basename(os.path.splitext(img_path)[0])
    
    for h in range(height // TARGET_PIXEL):
        for w in range(width // TARGET_PIXEL):
            patch_image = image[h*TARGET_PIXEL:(h+1)*TARGET_PIXEL, w*TARGET_PIXEL:(w+1)*TARGET_PIXEL]
            patch_mask = mask[(h*TARGET_PIXEL)//DOWNSTREAM:((h+1)*TARGET_PIXEL)//DOWNSTREAM, (w*TARGET_PIXEL)//DOWNSTREAM:((w+1)*TARGET_PIXEL)//DOWNSTREAM]
            patch_mask[np.where(patch_mask == 155)] = 0
            
            
            patch_mask = cv2.resize(patch_mask, (TARGET_PIXEL, TARGET_PIXEL), cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(SAVE_PATH, "images", f"{fname}_{w}_{h}.png"), patch_image)
            cv2.imwrite(os.path.join(SAVE_PATH, "masks", f"{fname}_{w}_{h}.png"), patch_mask)



if __name__ == "__main__":
    pairs = get_image_paths(r"/data01/notebooks/jh/skin")
    make_directory(os.path.join(SAVE_PATH, "images"))
    make_directory(os.path.join(SAVE_PATH, "masks"))

    for img_path, mask_path in pairs:
        print(img_path, mask_path)
        execute_patching(img_path, mask_path)


# In[ ]:




