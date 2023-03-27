import random
import torch

from monai.transforms import (AddChanneld, Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld)
from monai.data import DataLoader, Dataset, DistributedSampler, list_data_collate
from pretrain.datasets.utils import CategoricalToOneHot, SelectRelevantKeys
from fseft.datasets.utils import load_query
from utils.misc import seed_worker


def get_loader(args):

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=1, neg=1,
                               num_samples=args.num_samples, image_key="image", image_threshold=0),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        SelectRelevantKeys(),
        ToTensord(keys=["image", "label"]),
        CategoricalToOneHot(2),  # Foreground/Background
    ])

    # training dict part
    train_img, train_lbl, train_name = [], [], []
    for line in open(args.data_txt_path["train"]):

        train_img.append(args.data_root_path + line.strip().split()[0])
        train_lbl.append(args.data_root_path + line.strip().split()[1])
        train_name.append(line.strip().split()[1].split('.')[0])

    data_dicts_train = [{'image': image, 'label': label, 'name': name}
                        for image, label, name in zip(train_img, train_lbl, train_name)]

    # Randomly select k support samples
    random.seed(args.iFold-1)
    data_dicts_train = random.sample(data_dicts_train, args.k)
    args.data_dicts_train = data_dicts_train
    print('train len {}'.format(len(data_dicts_train)))

    # Set data loader
    train_dataset = Dataset(data=data_dicts_train, transform=transforms)
    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True, drop_last=True,
                                       seed=42) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42),
                              drop_last=True)

    # Load query sample
    load_query(args)

    return train_loader, train_sampler
