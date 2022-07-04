from __future__ import print_function, division
from re import M
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import cv2
import pdb
from pathlib import Path

def preprocess_test_data(final_test_mask_path, test_path, crop_test_path):
    bounding_boxes = {}

    for mask_name in os.listdir(final_test_mask_path):
        print('Process:', mask_name)
        img = cv2.imread(os.path.join(test_path, mask_name))
        H, W, _ = img.shape

        img_mask = cv2.imread(os.path.join(final_test_mask_path, mask_name))
        post_mask = np.moveaxis(img_mask, -1, 0)[0]
        post_mask = cv2.threshold(post_mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((5, 5), 'uint8')
        post_mask = cv2.erode(img_mask,kernel , iterations=8)
        post_mask = cv2.cvtColor(post_mask, cv2.COLOR_BGR2GRAY)
        post_mask = cv2.threshold(post_mask, 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(post_mask, 4, cv2.CV_32S)
        
        current_list_bbox = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area > 100000:
                #ignore background
                continue

            if area > 20000:
                xmin, ymin, xmax, ymax = x, y, x + w, y+h
                
                # Extending rectangle
                xmin = max(0, xmin - 100)
                xmax = min(W, xmax + 100)
                ymin = max(0, ymin - 100)
                ymax = min(H, ymax + 100)

                current_list_bbox.append([xmin, xmax, ymin, ymax])

        # sort in ascending order of first element in the list
        sorted_current_list = sorted(current_list_bbox, key=itemgetter(0))
        bounding_boxes[mask_name] = sorted_current_list
    
    # save new test images
    for img_name in os.listdir(test_path):
        img_read = cv2.imread(os.path.join(test_path, img_name))

        bbox = bounding_boxes[img_name]
        for idx_box, rec in enumerate(bbox):
            [xmin, xmax, ymin, ymax] = rec
            crop_test_path = os.path.join(crop_test_path, img_name.split('.')[0]+"__"+str(idx_box)+'.jpg')
            crop_chicken = img_read[int(ymin):int(ymax), int(xmin):int(xmax)]

            cv2.imwrite(crop_test_path, crop_chicken)

if __name__ == '__main__':
    final_test_mask_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/final_test_mask'
    crop_test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/classification/new_test/subfolder'
    test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/classification_data/test'
    
    Path(crop_test_path).mkdir(parents=True, exist_ok=True)
    
    # Crop chicken poses in a image
    preprocess_test_data(final_test_mask_path, test_path, crop_test_path)