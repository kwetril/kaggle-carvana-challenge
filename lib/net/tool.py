import random
import math
import cv2
import numpy as np


# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def randomHorizontalFlip2(image, label, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def randomShiftScaleRotate2(image, label, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1,0.1),
                            rotate_limit=(-45, 45), aspect_limit=(0, 0), borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if random.random() < u:
        height, width, channel = image.shape

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  #degree
        scale = random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * sx
        ss = math.sin(angle / 180 * math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0,))

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width, height])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width + 2 * dx, height + 2 * dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        height2, width2 = label.shape
        label = cv2.warpPerspective(label, mat, (width2, height2), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0,))
    return image, label
