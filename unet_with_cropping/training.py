import os
import sys
from timeit import default_timer as timer

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('..'))
from lib.net.segmentation.unet import UNet1024 as Net
from lib.net.segmentation.loss import SoftDiceLoss, BCELoss2d, dice_loss
from lib.net.tool import *
from lib.dataset import *


SEED = 5794
INPUT_HEIGHT = INPUT_WIDTH = 512
OUTPUT_HEIGHT = OUTPUT_WIDTH = 1024

SOLUTION_DATA_DIR = os.path.join(INPUT_PATH, "unet_with_cropping")
CHECKPOINT_DIR = os.path.join(SOLUTION_DATA_DIR, "checkpoint")
SNAPSHOT_DIR = os.path.join(SOLUTION_DATA_DIR, "snapshot")
LOGS_DIR = os.path.join(SOLUTION_DATA_DIR, "logs")
CROPPINGS_FILE_PATH = "./train_bboxes_with_padding_10.csv"


def shift_scale_transform(x, y):
    return randomShiftScaleRotate2(x, y, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0))


def hflip_transform(x, y):
    return randomHorizontalFlip2(x, y)


def criterion(logits, labels):
    loss = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)
    return loss


def run_train():
    initial_checkpoint = None
    # initial_checkpoint = os.path.join(CHECKPOINT_DIR, "10.pth")
    num_epoches = 10

    # setup directories
    os.makedirs(SOLUTION_DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "train.log")

    log = create_logger(log_path)
    log.info("Start training")
    log.info('Solution directory: %s' % SOLUTION_DATA_DIR)
    log.info("Initial checkpoint: %s" % initial_checkpoint)
    log.info('Random seed: %s' % SEED)

    # split data into train and validation parts, create datasets and dataloaders
    car_ids = sorted(list(set(x.split('_')[0] for x in os.listdir(TRAIN_DIR))))
    train_car_ids, validation_car_ids = train_test_split(car_ids, train_size=0.9, random_state=SEED)

    train_img_paths = [os.path.join(TRAIN_DIR, "%s_%02d.jpg" % (x, i))
                       for x in train_car_ids for i in range(1, NUM_VIEWS + 1)]
    train_label_paths = [os.path.join(TRAIN_LABEL_DIR, "%s_%02d_mask.gif" % (x, i))
                         for x in train_car_ids for i in range(1, NUM_VIEWS + 1)]
    validation_img_paths = [os.path.join(TRAIN_DIR, "%s_%02d.jpg" % (x, i))
                            for x in validation_car_ids for i in range(1, NUM_VIEWS + 1)]
    validation_label_paths = [os.path.join(TRAIN_LABEL_DIR, "%s_%02d_mask.gif" % (x, i))
                              for x in validation_car_ids for i in range(1, NUM_VIEWS + 1)]

    batch_size = 4
    img_croppings = read_croppings(CROPPINGS_FILE_PATH)
    train_dataset = CroppedCarDataset(train_img_paths, img_croppings, img_resize=(INPUT_HEIGHT, INPUT_WIDTH),
                                      label_paths=train_label_paths, label_resize=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
                                      transform=[shift_scale_transform, hflip_transform], is_preload=False)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                              batch_size=batch_size, drop_last=True, num_workers=4, pin_memory=True)

    valid_dataset = CroppedCarDataset(validation_img_paths, img_croppings, img_resize=(INPUT_HEIGHT, INPUT_WIDTH),
                                      label_paths=validation_label_paths, label_resize=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
                                      is_preload=False)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                              batch_size=batch_size, drop_last=False, num_workers=2, pin_memory=True)

    log.info('batch_size          = %d' % batch_size)
    log.info('len(train_dataset)  = %d' % len(train_dataset))
    log.info('len(valid_dataset)  = %d' % len(valid_dataset))

    net = Net(in_shape=(3, INPUT_HEIGHT, INPUT_WIDTH), num_classes=1)
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    log.info('Network type: %s' % type(net))

    # change to skip checkpoints saving for some epochs
    epoch_save = [x for x in range(num_epoches + 1)]

    # resume learning process from checkpoint
    start_epoch=0
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['state_dict'])

    # training
    log.info('epoch    iter      rate   | smooth_loss/acc | train_loss/acc | valid_loss/acc ...')
    log.info('---------------------------------------------------------------------------------')

    smooth_loss = 0.0
    smooth_acc = 0.0
    train_loss = np.nan
    train_acc = np.nan
    for epoch in range(start_epoch, num_epoches):
        epoch_start = timer()

        # block to adjust learning rate
        if epoch >= 25:
            adjust_learning_rate(optimizer, lr=0.005)
        if epoch >= 60:
            adjust_learning_rate(optimizer, lr=0.004)
        if epoch >= 80:
            adjust_learning_rate(optimizer, lr=0.002)
        if epoch >= num_epoches - 5:
            adjust_learning_rate(optimizer, lr=0.001)

        rate = get_learning_rate(optimizer)[0]

        sum_smooth_loss = 0.0
        sum_smooth_acc = 0.0
        num_batches = 0
        net.train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images.cuda())
            labels  = Variable(labels.cuda())

            # forward pass
            logits = net(images)
            probs = F.sigmoid(logits)
            masks = (probs>0.5).float()

            # backward pass
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss / score values for learning process monitoring
            acc = dice_loss(masks, labels)

            sum_smooth_loss += loss.data[0]
            sum_smooth_acc += acc .data[0]
            num_batches += 1

            smooth_loss = sum_smooth_loss / num_batches
            smooth_acc = sum_smooth_acc / num_batches

            train_acc = acc.data[0]
            train_loss = loss.data[0]
            log.info('%5.2f   %5d    %0.5f   |  %0.5f  %0.5f | %0.5f  %6.5f | ...' %
                     (epoch + (it + 1) / num_its, it + 1, rate, smooth_loss, smooth_acc, train_loss, train_acc))

        epoch_end = timer()
        time = (epoch_end - epoch_start) / 60

        # validation
        net.eval()
        valid_loss, valid_acc = predict_and_evaluate(net, valid_loader)

        log.info('%5.2f   %5d    %0.5f   |  %0.5f  %0.5f | %0.5f  %6.5f | %0.5f  %6.5f  |  %3.1f min' % \
                 (epoch + 1, it + 1, rate, smooth_loss, smooth_acc, train_loss, train_acc, valid_loss, valid_acc, time))

        if epoch in epoch_save:
            torch.save(net.state_dict(), os.path.join(SNAPSHOT_DIR, '%03d.pth' % epoch))
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(CHECKPOINT_DIR, '%03d.pth' % epoch))

    # save final model
    log.info("Save final model")
    torch.save(net.state_dict(), os.path.join(SNAPSHOT_DIR, 'final.pth'))
    log.info('Success!')


def predict_and_evaluate(net, test_loader):
    """
    Calculate loss and score on validation part of dataset
    """

    test_acc = 0
    test_loss = 0
    test_num = 0
    for it, (images, labels, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        # forward
        logits = net(images)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()

        loss = criterion(logits, labels)
        acc = dice_loss(masks, labels)

        batch_size = len(indices)
        test_num += batch_size
        test_loss += batch_size * loss.data[0]
        test_acc += batch_size * acc.data[0]
    assert(test_num == len(test_loader.sampler))

    test_loss = test_loss / test_num
    test_acc = test_acc / test_num
    return test_loss, test_acc


if __name__ == '__main__':
    run_train()
