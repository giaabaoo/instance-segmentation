import os
import os.path as osp
import cv2
import numpy as np
import copy
import pdb
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    test_mask_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/test_mask'
    final_test_mask_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/final_test_mask'

    Path(final_test_mask_path).mkdir(parents = True, exist_ok=True)


    for test_mask in tqdm(os.listdir(test_mask_path)):
        mask_path = osp.join(test_mask_path, test_mask)
        img_mask = cv2.imread(mask_path)

        kernel = np.ones((5, 5), 'uint8')
        post_mask = cv2.erode(img_mask,kernel , iterations=8)
        post_mask = cv2.cvtColor(post_mask, cv2.COLOR_BGR2GRAY)
        post_mask = cv2.threshold(post_mask, 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(post_mask, 4, cv2.CV_32S)
       
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # ensure area are not too small
            if area < 10000:
                labels[y:y+h, x:x+w] = 0
        
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0
        
        # save img
        cv2.imwrite(osp.join(final_test_mask_path, test_mask), labeled_img)