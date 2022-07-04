from mmseg.apis import set_random_seed, inference_segmentor, init_segmentor
from dataset import ChickenDataset
from mmconfig import cfg

import os
import os.path as osp
import numpy as np
import pdb
import cv2
from tqdm import tqdm
import torch.distributed as dist
from pathlib import Path
PALETTE = [[255, 255, 255], [0, 0, 0]]

model = init_segmentor(cfg, '/home/nttung/BB/Instance_Semantic_Segmentation/mmsegmentation/custom_train/work_dirs/tutorial/latest.pth', device='cuda:0')

model.cfg = cfg


'''Inference on chicken test data'''

data_test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/test'
data_test_mask_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/test_mask'
Path(data_test_mask_path).mkdir(parents = True, exist_ok=True)

for img_test in tqdm(os.listdir(data_test_path)):
    img_test_path = osp.join(data_test_path, img_test)
    result = inference_segmentor(model, img_test_path)

    result = np.rollaxis(np.array(result), 0 , 3)
    result *= 255

    cv2.imwrite(osp.join(data_test_mask_path, img_test), result)

    # model.show_result(img, result, out_file=osp.join(data_test_mask_path, img_test), palette=PALETTE)