import sys, os
sys.path.insert(0, os.path.abspath('..'))
from lib.utils import *

import numpy as np
import glob
import cv2
from PIL import Image


def get_avg_mask(img_num):
    '''
    Calculates average mask for images with given number (rotation)
    and then searches for best threshold to binarize the mask.
    Based on kernel: https://www.kaggle.com/zusmani/baseline-optimal-mask
    '''

    img_num = str(img_num) if img_num >= 10 else "0{}".format(img_num)
    train_files = glob.glob(os.path.join(INPUT_PATH, "train_masks", "*_{}_mask.gif").format(img_num))
    train_masks = []    
    avg_mask = np.zeros((1280, 1918), dtype=np.float64)
    for f in train_files:
        mask = np.array(Image.open(f), dtype=np.uint8)       
        train_masks.append(mask)
        avg_mask += mask.astype(np.float64)
    avg_mask /= len(train_files)
    train_masks = np.array(train_masks, dtype=np.uint8)
    print(avg_mask.min(), avg_mask.max(), train_masks.shape)

    best_score = 0
    best_thr = -1
    for t in range(410, 460, 5):
        thr = t/1000
        avg_mask_thr = avg_mask.copy()
        avg_mask_thr[avg_mask_thr > thr] = 1
        avg_mask_thr[avg_mask_thr <= thr] = 0
        score = get_score(train_masks, [avg_mask_thr for x in train_masks])
        print('NUM: {} THR: {:.3f} SCORE: {:.6f}'.format(img_num, thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr

    print('{}: Best score: {} Best thr: {}'.format(img_num, best_score, best_thr))
    avg_mask_thr = avg_mask.copy()
    avg_mask_thr[avg_mask_thr > best_thr] = 1
    avg_mask_thr[avg_mask_thr <= best_thr] = 0
    avg_mask_thr[avg_mask_thr > 0.5] = 1
    avg_mask_thr[avg_mask_thr <= 0.5] = 0
    print(avg_mask.shape, avg_mask_thr.shape)
    cv2.imwrite('avg_mask_{}.jpg'.format(img_num), (255*avg_mask_thr).astype(np.uint8))
    return best_score, avg_mask_thr

    
def predict_masks(masks, test_files):
    for test_file in test_files:
        yield (test_file, masks[get_num(test_file)])


if __name__ == '__main__':
    masks = [None]
    scores = []
    for i in range(1, 17):
        best_score, avg_mask = get_avg_mask(i)
        masks.append(avg_mask)
        scores.append(best_score)
    best_score = sum(scores) / len(scores)
    print("Best score: {}".format(best_score))
    test_imgs = os.listdir(TEST_DIR)
    create_submission(predict_masks(masks, test_imgs))