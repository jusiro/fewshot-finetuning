import torch
import numpy as np

from utils.metrics import extract_topk_largest_candidates
from monai.transforms import (LoadImaged, AddChanneld, Compose, CropForegroundd, Orientationd, ScaleIntensityRanged,
                              Spacingd, ScaleIntensityRangePercentilesd)
from pretrain.datasets.utils import SelectAdaptationOrgan, select_intensity_scaling

from local_data.constants import PATH_DATASETS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_query(args):

    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            SelectAdaptationOrgan(args),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(args.model_cfg["space_x"], args.model_cfg["space_y"],
                                                      args.model_cfg["space_z"]), mode=("bilinear", "nearest")),
            select_intensity_scaling(args),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    # Train dict part
    test_samples = 0
    img, lbl, name = [], [], []
    for iPartition in ['test']:
        for line in open(args.data_txt_path[iPartition]):
            item = line.strip().split()[0].split('/')[1]

            img.append(PATH_DATASETS + line.strip().split()[0])
            lbl.append(PATH_DATASETS + line.strip().split()[1])
            name.append(line.strip().split()[1].split('.')[0])

            test_samples += 1

    data_dicts = [{'image': image, 'label': label, 'name': name}
                  for image, label, name in zip(img, lbl, name)]

    # Get sample info
    dict_data = data_dicts[args.iFold-1]

    # Load image and labels
    dict_data = transforms(dict_data)

    image = dict_data['image']
    if args.objective == "binary":
        label = (dict_data['label'] == 1).astype(int)
        label = np.expand_dims(extract_topk_largest_candidates(np.squeeze(label), 1), 0)
    else:
        label = dict_data['label'].astype(int)

    # Select image id name
    if "TotalSegmentator" in data_dicts[args.iFold-1]["name"]:
        id_test = data_dicts[args.iFold - 1]["name"].split("/")[-3]
    else:
        id_test = data_dicts[args.iFold - 1]["name"].split("/")[-1]

    args.query = {'image': image, 'label': label, "name": id_test}