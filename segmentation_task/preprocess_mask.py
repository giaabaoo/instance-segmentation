import os.path as osp 
import cv2
import pdb
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
palette = [[255, 255, 255], [0,0, 0]]

if __name__ == '__main__':
    mask_path = "/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/train/mask"
    new_mask_path = "/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/train/grayscale_mask"

    Path(new_mask_path).mkdir(parents=True, exist_ok=True)

    for img in tqdm(os.listdir(mask_path)):
        image = Image.open(osp.join(mask_path, img)).convert('RGB')
        custom_grayscale = image.convert('P')
        custom_grayscale.putpalette(np.array(palette, dtype=np.uint8))
        pdb.set_trace()
        custom_grayscale.save(osp.join(new_mask_path, img.replace('.jpg', '.png')))
