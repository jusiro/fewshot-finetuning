import torch
import numpy as np

from utils.metrics import extract_topk_largest_candidates
from monai.transforms import (LoadImaged, AddChanneld, Compose, CropForegroundd, Orientationd, ScaleIntensityRanged,
                              Spacingd)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_query(args):

    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    # Train dict part
    test_samples = 0
    img, lbl, name = [], [], []
    for iPartition in ['test']:
        for line in open(args.data_txt_path[iPartition]):
            item = line.strip().split()[0].split('/')[1]

            img.append(args.data_root_path + line.strip().split()[0])
            lbl.append(args.data_root_path + line.strip().split()[1])
            name.append(item)

            test_samples += 1

    data_dicts = [{'image': image, 'label': label, 'name': name}
                  for image, label, name in zip(img, lbl, name)]
    print('Dataset len {}'.format(len(data_dicts)))

    # Get sample info
    dict_data = data_dicts[args.iFold-1]

    # Load image and labels
    dict_data = transforms(dict_data)

    image = dict_data['image']
    label = (dict_data['label'] == 1).astype(int)
    label = np.expand_dims(extract_topk_largest_candidates(np.squeeze(label), 1), 0)

    args.query = {'image': image, 'label': label}