import torch
import cc3d
import fastremap
import numpy as np


def dice_score(preds, labels):  # on GPU

    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    return dice, recall, precision


def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()
    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label
