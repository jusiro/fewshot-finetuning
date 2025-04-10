from monai.transforms import (AddChanneld, Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld)
from monai.data import DataLoader, Dataset, DistributedSampler, list_data_collate
from pretrain.datasets.utils import CategoricalToOneHot, SelectRelevantKeys, UNIVERSAL_TEMPLATE, MapLabels, \
    LRDivision, UniformDataset, RandZoomd_select
import random


def get_loader(args):

    # Prepare data transforms and augmentations
    if args.stage == 'train':
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            SelectRelevantKeys(),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                     mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                                 clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            LRDivision(),
            SelectRelevantKeys(),
            RandZoomd_select(keys=["image", "label"], prob=0.3, min_zoom=1.1, max_zoom=1.3, mode=['area', 'nearest']),
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                   spatial_size=(args.roi_x, args.roi_y, args.roi_z), pos=4, neg=1,
                                   num_samples=args.num_samples, image_key="image", image_threshold=0),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
            SelectRelevantKeys(),
            MapLabels(),
            ToTensord(keys=["image", "label"]),
            CategoricalToOneHot(args.NUM_CLASS+1),
        ])
    elif args.stage == 'val':
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
            SelectRelevantKeys(),
            MapLabels(),
            ToTensord(keys=["image", "label"]),
            CategoricalToOneHot(len(UNIVERSAL_TEMPLATE.keys())+1),
        ])

    # training dict part
    train_img, train_lbl, train_name = [], [], []
    for iPartition in args.partitions:
        for line in open(args.data_txt_path[iPartition]):

            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_name.append(line.strip().split()[1].split('.')[0])

    data_dicts_train = [{'image': image, 'label': label, 'name': name}
                        for image, label, name in zip(train_img, train_lbl, train_name)]

    print('train len {}'.format(len(data_dicts_train)))

    if args.balanced:
        train_dataset = UniformDataset(data=data_dicts_train, transform=transforms)
    else:
        train_dataset = Dataset(data=data_dicts_train, transform=transforms)

    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=args.shuffle) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers, collate_fn=list_data_collate)

    return train_loader, train_sampler
