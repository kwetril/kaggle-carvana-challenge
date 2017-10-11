import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from lib.net.segmentation.unet import UNet1024 as Net
from lib.net.tool import *
from lib.dataset import *
from lib.utils import *
import multiprocessing

from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn.functional as F
from torch.autograd import Variable

INPUT_HEIGHT = INPUT_WIDTH = 512
OUTPUT_HEIGHT = OUTPUT_WIDTH = 1024
CROPPINGS_FILE_PATH = "./test_bboxes_with_padding_10.csv"


def predict(net, test_loader, names, img_croppings, flip=True):
    """
    For each batch from :param test_loader: make prediction of the masks using :param net: model
    and for each image in batch generate pair (img_name, rle_mask)
    """

    test_dataset = test_loader.dataset
    test_iter = iter(test_loader)
    test_num = len(test_dataset)
    batch_size = test_loader.batch_size

    num = 0
    threshold = 0.5 * 255
    # iterate over batches
    for m in range(0, test_num, batch_size):
        images, indices = test_iter.next()
        if images is None:
            break

        flipped_images = torch.from_numpy(np.flip(images.numpy(), 3).copy())

        # make prediction for batch
        images = Variable(images.cuda(), volatile=True)
        logits = net(images)
        probs = F.sigmoid(logits)
        probs = probs.data.cpu().numpy().reshape(-1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        if flip:
            # make prediction for flipped batch and calculate average
            flipped_images = Variable(flipped_images.cuda(), volatile=True)
            flipped_logits = net(flipped_images)
            flipped_probs = F.sigmoid(flipped_logits)
            flipped_probs = flipped_probs.data.cpu().numpy().reshape(-1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
            unflipped_probs = np.flip(flipped_probs, 2)
            probs = (probs + unflipped_probs) / 2 * 255

        # process batch predictions, generate answers
        for p in probs:
            bbox = img_croppings[names[num]]
            top, left, bottom, right = bbox
            p = cv2.resize(p, (right - left, bottom - top))
            mask = np.zeros((INIT_HEIGHT, INIT_WIDTH), dtype=np.uint8)
            mask[top:bottom, left:right] = p > threshold
            yield (names[num] + '.jpg', mask)
            num += 1
    assert(test_num == num)


def run_predict(model_file):
    model_path = os.path.join(MODELS_PATH, model_file)
    batch_size = 4

    test_img_paths = np.array([os.path.join(TEST_DIR, x) for x in sorted(os.listdir(TEST_DIR))])
    img_croppings = read_croppings(CROPPINGS_FILE_PATH)
    test_dataset = CroppedCarDataset(test_img_paths, img_croppings,
                                     img_resize=(INPUT_HEIGHT, INPUT_WIDTH), is_preload=False)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size,
                             drop_last=False, num_workers=2, pin_memory=True)
    names = [os.path.basename(x).split('.')[0] for x in test_dataset.img_paths]

    net = Net(in_shape=(3, INPUT_HEIGHT, INPUT_WIDTH), num_classes=1)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    create_submission(predict(net, test_loader, names, img_croppings))


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    run_predict('unet_125_wider_with_crop.pth')
    print('\nsucess!')
