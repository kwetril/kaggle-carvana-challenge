import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import PIL.Image
from lib.dataset import *
from lib.utils import *
import matplotlib.pyplot as plt


def read_train_mask_as_np(filename):
    """
    :param filename: file name of one of the train masks '00087a6bd4dc_01_mask.gif'
    :return: mask as numpy array
    """
    label_path = os.path.join(TRAIN_LABEL_DIR, filename)
    img = np.array(PIL.Image.open(label_path))
    return img


def get_mask_bounding_box(mask):
    """
    :param mask: mask as numpy array
    :return: tuple with mask bounds (top, left, bottom, right)
    """
    nonzero_idxs = np.argwhere(mask)
    top = np.min(nonzero_idxs[:, 0])
    left = np.min(nonzero_idxs[:, 1])
    bottom = np.max(nonzero_idxs[:, 0])
    right = np.max(nonzero_idxs[:, 1])
    return (top, left, bottom + 1, right + 1)


def compute_train_bboxes():
    """
    Creates file with description of minimal bounding boxes for train images.
    """
    train_label_files = os.listdir(TRAIN_LABEL_DIR)
    with open("train_bboxes.csv", "w") as out:
        for train_label_file in train_label_files:
            mask = read_train_mask_as_np(train_label_file)
            bbox = get_mask_bounding_box(mask)
            out.write("{}\t{}\t{}\t{}\t{}\n".format(train_label_file, bbox[0], bbox[1], bbox[2], bbox[3]))


def compute_train_bboxes_with_padding(padding=10):
    """
    Uses minimal bounding boxes to compute bounding boxes with padding and saves their description into new file.
    :param padding: size of padding around minimal bounding box
    """
    with open("train_bboxes.csv", "r") as input:
        with open("train_bboxes_with_padding_{}.csv".format(padding), "w") as out:
            for line in input:
                parts = line.split("\t")
                mask_file = parts[0]
                top = int(parts[1])
                left = int(parts[2])
                bottom = int(parts[3])
                right = int(parts[4])
                top = np.maximum(0, top - padding)
                left = np.maximum(0, left - padding)
                bottom = np.minimum(INIT_HEIGHT, bottom + padding)
                right = np.minimum(INIT_WIDTH, right + padding)
                out.write("{}\t{}\t{}\t{}\t{}\n".format(mask_file, top, left, bottom, right))


def compute_test_bboxes():
    """
    Creates file with description of minimal bounding boxes for test images
    taking submission file with rle masks as input.
    """
    submission_file_path = os.path.join("..", "submissions", "sample_submission.csv")
    is_first = True
    i = 0
    with open(submission_file_path, "r") as rles:
        with open("test_bboxes.csv", "w") as out:
            for line in rles:
                if is_first:
                    is_first = False
                    continue
                i += 1
                if i % 10000 == 0:
                    print(i)
                parts = line.split(',', 1)
                mask = rle_decode(parts[1], INIT_HEIGHT, INIT_WIDTH, 1)
                bbox = get_mask_bounding_box(mask)
                out.write("{}\t{}\t{}\t{}\t{}\n".format(parts[0], bbox[0], bbox[1], bbox[2], bbox[3]))


def compute_test_bbox_with_paddings(padding=10):
    """
    Uses minimal bounding boxes to compute bounding boxes with padding and saves their description into new file.
    :param padding: size of padding around minimal bounding box
    """
    with open("test_bboxes.csv", "r") as input:
        with open("test_bboxes_with_padding_{}.csv".format(padding), "w") as out:
            for line in input:
                parts = line.split("\t")
                mask_file = parts[0]
                top = int(parts[1])
                left = int(parts[2])
                bottom = int(parts[3])
                right = int(parts[4])
                top = np.maximum(0, top - padding)
                left = np.maximum(0, left - padding)
                bottom = np.minimum(INIT_HEIGHT, bottom + padding)
                right = np.minimum(INIT_WIDTH, right + padding)
                out.write("{}\t{}\t{}\t{}\t{}\n".format(mask_file, top, left, bottom, right))


def show_test_bbox(file_name):
    """
    :param file_name: name of test image to show bounding box for.
    """
    items = {}
    with open("test_bboxes.csv", "r") as bboxes:
        for line in bboxes:
            parts = line.split('\t')
            items[parts[0]] = tuple(parts[1:])
    bbox = items[file_name]
    path = os.path.join(TEST_DIR, file_name)
    img = cv2.imread(path)
    top = int(bbox[0])
    left = int(bbox[1])
    bottom = int(bbox[2])
    right = int(bbox[3])
    plt.imshow(img[top:bottom, left:right])
    plt.show()


if __name__ == "__main__":
    compute_train_bboxes()
    # compute_train_bboxes_with_padding()
    # compute_test_bboxes()
    # compute_test_bbox_with_paddings()
    # show_test_bbox("0004d4463b50_01.jpg")
