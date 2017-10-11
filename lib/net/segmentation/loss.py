import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
# https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)


# https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
# https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score


# https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2):
    num = m1.size(0)
    m1 = m1.view(num, -1)
    m2 = m2.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num
    return score


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    """Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: True

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        logits = logits.view(num, -1)
        targets = targets.view(num, -1)
        score = self.focal_loss(logits, targets)
        return score

    def focal_loss(self, x, y):
        """
        Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        """
        alpha = 0.5
        gamma = 2

        t = y

        p = x.sigmoid()
        pt = p.clone()
        pt[t == 0] = 1 - p[t == 0]  # pt = p if t>0 else 1-p

        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return binary_cross_entropy_with_logits(x, t, w, size_average=True)
