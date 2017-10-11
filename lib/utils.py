from datetime import datetime
import numpy as np
import pandas as pd
import os
import logging
from logging.config import dictConfig
from os.path import join, abspath, normpath, basename

INPUT_PATH = join('..', 'data')
TRAIN_DIR = join(INPUT_PATH, 'train')
TRAIN_LABEL_DIR = join(INPUT_PATH, 'train_masks')
TEST_DIR = join(INPUT_PATH, 'test')
MODELS_PATH = join(INPUT_PATH, 'models')
INIT_HEIGHT = 1280
INIT_WIDTH = 1918
NUM_VIEWS = 16


def rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Taken from https://www.kaggle.com/stainsby/fast-tested-rle
    """
    pixels = img.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H * W), np.uint8)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0] - 1
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(H, W)
    return mask
    

def dice(im1, im2, empty_score=1.0):
    """
    Metric to optimize
    """
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape. im1: {}; im2: {}".format(
        im1.shape, im2.shape))

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum
    

def get_score(true_masks, predicted_masks):
    """
    Calculate score of predictions
    """
    d = 0.0
    for i in range(true_masks.shape[0]):
        d += dice(true_masks[i], predicted_masks[i])
    return d / true_masks.shape[0]
    

def get_id(name):
    return name.split("_")[0]
    

def get_num(name):
    return int(name.split("_")[1].split('.')[0])


def get_img_name(img_path):
    filename = os.path.basename(img_path)
    if filename.endswith("_mask.gif"):
        return filename[:len(filename) - len("_mask.gif")]
    else:
        return filename[:len(filename) - len(".jpg")]

    
def create_submission(imgs_with_mask_generator):
    """
    Prepare submission file.
    imgs - list of image file names
    masks - corresponding predicted masks for images
    """
    print('Create submission...')
    os.makedirs(os.path.join("..", 'submissions'), exist_ok=True)
    date_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = basename(normpath((abspath('.'))))
    out_file = "{}_{}.csv.gz".format(date_part, parent_dir)
    out_path = join("..", "submissions", out_file)
    template = pd.read_csv(join(INPUT_PATH, 'sample_submission.csv'))
    i = 0
    for img, mask in imgs_with_mask_generator:
        row = template.iloc[i]
        row['rle_mask'] = rle(mask)
        row['img'] = img
        if i % 1000 == 0:
            print(i)
        i += 1
    template.to_csv(out_path, index=False, compression='gzip')
    print("Done: {}".format(out_path))


def create_logger(log_path):
    logging_config = dict(
        version=1,
        formatters={
            'f': {
                'format': '%(asctime)s %(name)-10s %(levelname)-8s %(message)s'
            }
        },
        handlers={
            'h': {'class': 'logging.handlers.RotatingFileHandler',
                  'filename': log_path,
                  'backupCount': 10,
                  'maxBytes': 5 * 1024 * 1024,
                  'formatter': 'f',
                  'level': logging.DEBUG
            },
            'ch': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'level': logging.DEBUG
            }
        },
        loggers={
            'root': {
                'handlers': ['h', 'ch'],
                'level': logging.DEBUG
            }
        }
    )
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('root')
    return logger


def read_croppings(file_path):
    res = {}
    with open(file_path) as f:
        for line in f:
            parts = line.split("\t")
            res[get_img_name(parts[0])] = tuple([int(x) for x in parts[1:]])
    return res
