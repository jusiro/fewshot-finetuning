import torch


class BinaryDice3D(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(BinaryDice3D, self).__init__()
        self.reduction = reduction
        self.smooth = 1

    def forward(self, predict, target, annotation_mask=None):
        num = torch.sum(predict*target, (2, 3, 4))
        den = torch.sum(predict, (2, 3, 4)) + torch.sum(target, (2, 3, 4))

        dice_score = ((2 * num) + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice_score

        if annotation_mask is not None:
            dice_loss = torch.mean(torch.sum(dice_loss * annotation_mask, -1) / torch.sum(annotation_mask, -1))
        else:
            dice_loss = dice_loss.mean()

        return dice_loss
