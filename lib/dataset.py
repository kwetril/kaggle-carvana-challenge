import PIL.Image
import cv2
import torch
from torch.utils.data.dataset import Dataset

from lib.utils import *


class CarDataset(Dataset):
    def __init__(self, img_paths, img_resize=None, label_paths=None, label_resize=None, transform=[], is_preload=True):
        channel = 3
        self.img_height, self.img_width = (INIT_HEIGHT, INIT_WIDTH) if img_resize is None else img_resize
        self.label_height, self.label_width = (INIT_HEIGHT, INIT_WIDTH) if label_resize is None else label_resize

        self.img_paths = img_paths
        self.img_resize = img_resize
        self.label_paths = label_paths
        self.label_resize = label_resize
        self.transform = transform

        num_images = len(img_paths)

        images = None
        if is_preload:
            images = np.zeros((num_images, self.img_height, self.img_width, channel), dtype=np.float32)
            for i in range(num_images):
                images[i] = self.read_img(img_paths[i])

        labels = None
        if is_preload and label_paths is not None:
            assert len(label_paths) == num_images, "Number of labels {} not equal to number of images {}".format(
                len(label_paths), num_images)
            labels = np.zeros((num_images, self.label_height, self.label_width), dtype=np.float32)
            for i in range(num_images):
                labels[i] = self.read_label(label_paths[i])

        self.images = images
        self.labels = labels

    def read_img(self, path):
        img = cv2.imread(path)
        if self.img_resize is not None:
            img = cv2.resize(img, (self.img_width, self.img_height))
        return img / 255.

    def read_label(self, path):
        mask = np.array(PIL.Image.open(path))
        if self.label_resize is not None:
            mask = cv2.resize(mask, (self.label_width, self.label_height))
        return mask

    # https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        image = self.read_img(self.img_paths[index]) if self.images is None else self.images[index]

        if self.label_paths is None:
            for t in self.transform:
                image = t(image)
            image = image_to_tensor(image)
            return image, index
        else:
            label = self.read_label(self.label_paths[index]) if self.labels is None else self.labels[index]
            for t in self.transform:
                image, label = t(image, label)
            image = image_to_tensor(image)
            label = label_to_tensor(label)
            return image, label, index

    def __len__(self):
        return len(self.img_paths)


class CroppedCarDataset(Dataset):
    def __init__(self, img_paths, img_croppings, img_resize=None, label_paths=None, label_resize=None, transform=[], is_preload=True):
        channel = 3
        self.img_height, self.img_width = (INIT_HEIGHT, INIT_WIDTH) if img_resize is None else img_resize
        self.label_height, self.label_width = (INIT_HEIGHT, INIT_WIDTH) if label_resize is None else label_resize

        self.img_paths = img_paths
        self.img_croppings = img_croppings
        self.img_resize = img_resize
        self.label_paths = label_paths
        self.label_resize = label_resize
        self.transform = transform

        num_images = len(img_paths)

        images = None
        if is_preload:
            images = np.zeros((num_images, self.img_height, self.img_width, channel), dtype=np.float32)
            for i in range(num_images):
                images[i] = self.read_img(img_paths[i])

        labels = None
        if is_preload and label_paths is not None:
            assert len(label_paths) == num_images, "Number of labels {} not equal to number of images {}".format(
                len(label_paths), num_images)
            labels = np.zeros((num_images, self.label_height, self.label_width), dtype=np.float32)
            for i in range(num_images):
                labels[i] = self.read_label(label_paths[i])

        self.images = images
        self.labels = labels

    def read_img(self, path):
        img = cv2.imread(path)
        bbox = self.img_croppings[get_img_name(path)]
        top, left, bottom, right = bbox
        img = img[top:bottom, left:right]
        if self.img_resize is not None:
            img = cv2.resize(img, (self.img_width, self.img_height))
        return img / 255.

    def read_label(self, path):
        mask = np.array(PIL.Image.open(path))
        bbox = self.img_croppings[get_img_name(path)]
        top, left, bottom, right = bbox
        mask = mask[top:bottom, left:right]
        if self.label_resize is not None:
            mask = cv2.resize(mask, (self.label_width, self.label_height))
        return mask

    # https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        image = self.read_img(self.img_paths[index]) if self.images is None else self.images[index]

        if self.label_paths is None:
            for t in self.transform:
                image = t(image)
            image = image_to_tensor(image)
            return image, index
        else:
            label = self.read_label(self.label_paths[index]) if self.labels is None else self.labels[index]
            for t in self.transform:
                image, label = t(image, label)
            image = image_to_tensor(image)
            label = label_to_tensor(label)
            return image, label, index

    def __len__(self):
        return len(self.img_paths)


def image_to_tensor(image, mean=0, std=1.):
    """
    transform (input is numpy array, read in by cv2)
    """
    image = image.astype(np.float32)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def label_to_tensor(label, threshold=0.5):
    label = label
    label = (label > threshold).astype(np.float32)
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor
