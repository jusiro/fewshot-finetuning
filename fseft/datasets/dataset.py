import random
import torch

from monai.transforms import (AddChanneld, Compose, CropForegroundd, Orientationd, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, ToTensord, SpatialPadd, LoadImaged,
                              RandCropByPosNegLabeld, ScaleIntensityRangePercentilesd)
from monai.data import DataLoader, Dataset, list_data_collate

from pretrain.datasets.utils import CategoricalToOneHot, SelectRelevantKeys, SelectAdaptationOrgan,\
    select_intensity_scaling
from fseft.datasets.utils import load_query
from utils.misc import seed_worker

from local_data.constants import PATH_DATASETS


def get_loader(args):

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        SelectAdaptationOrgan(args),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.model_cfg["space_x"], args.model_cfg["space_y"],
                                                  args.model_cfg["space_z"]),
                 mode=("bilinear", "nearest")),
        select_intensity_scaling(args),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.model_cfg["roi_x"], args.model_cfg["roi_y"],
                                                           args.model_cfg["roi_z"]), mode='constant'),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(args.model_cfg["roi_x"], args.model_cfg["roi_y"],
                                             args.model_cfg["roi_z"]), pos=1, neg=1,
                               num_samples=args.adapt_hp["num_samples"], image_key="image", image_threshold=0),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        SelectRelevantKeys(),
        ToTensord(keys=["image", "label"]),
        CategoricalToOneHot(args),  # Background + classes
    ])

    # training dict part
    train_img, train_lbl, train_name = [], [], []
    for line in open(args.data_txt_path["train"]):

        train_img.append(PATH_DATASETS + line.strip().split()[0])
        train_lbl.append(PATH_DATASETS + line.strip().split()[1])
        train_name.append(line.strip().split()[1].split('.')[0])

    data_dicts = [{'image': image, 'label': label, 'name': name}
                  for image, label, name in zip(train_img, train_lbl, train_name)]

    # Randomly select k support samples
    random.seed(args.iFold-1)
    data_dicts = random.sample(data_dicts, min(args.k + 1, 30))

    # Leave one sample out to eval
    data_dicts_train = data_dicts[:-1]
    data_dicts_val = data_dicts[-1]

    args.data_dicts_train = data_dicts_train
    print('train len {}'.format(len(data_dicts_train)))

    # Set training data loader
    train_dataset = Dataset(data=data_dicts_train, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.adapt_hp["batch_size"], shuffle=True,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42),
                              drop_last=True)

    # Load validation batch
    if args.early_stop_criteria == "val":
        data_val = transforms(data_dicts_val)
        data_val_dict = {"image": torch.cat([sample["image"].unsqueeze(0) for sample in data_val], 0),
                         "label": torch.cat([sample["label"].unsqueeze(0) for sample in data_val], 0)}
        args.data_val_dict = data_val_dict

    # Load query sample for testing
    load_query(args)

    return train_loader
