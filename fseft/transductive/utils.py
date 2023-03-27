import torch
import numpy as np

from monai.transforms import (LoadImaged, AddChanneld, Compose, CropForegroundd, Orientationd, ScaleIntensityRanged,
                              Spacingd, ToTensord)
from utils.metrics import extract_topk_largest_candidates

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_query(args):
    print("Preprocessing query")

    x_query = args.query['image']
    x_query = np.pad(x_query, ((0, 0),
                               (0, args.roi_x - x_query.shape[1] % args.roi_x),
                               (0, args.roi_x - x_query.shape[2] % args.roi_x),
                               (0, args.roi_x - x_query.shape[3] % args.roi_x)))
    x_query = torch.tensor(x_query, dtype=torch.float32).to(device)
    x_query = x_query.unfold(
        1, args.roi_x, args.roi_x).unfold(
        2, args.roi_x, args.roi_x).unfold(
        3, args.roi_x, args.roi_x).reshape(-1, args.roi_x, args.roi_x, args.roi_x).unsqueeze(1)
    x_query_batched = x_query.cpu().numpy()

    return x_query_batched


def store_features(x_query_batched, model, scale=2):
    print("Extracting features from query sample")
    x_query = torch.tensor(x_query_batched, dtype=torch.float32).to(device)

    bs, idx_, feats_query_batched = 16, 0, []
    model.module.out_features = True
    while idx_ < x_query.shape[0]:
        # Select minibatch with patches
        x_patch = x_query[idx_:min(idx_ + bs, x_query.shape[0]), :, :, :]

        if len(x_patch.shape) < 5:
            x_patch = x_patch.unsqueeze(0)

        with torch.no_grad():
            # Extract features
            feats = model(x_patch)
            # Reduce dimension due to memory constraints
            feats = torch.nn.AvgPool3d(scale)(feats)

        # Store
        feats_query_batched.append(feats.detach().cpu().numpy())

        # Clear memory
        torch.cuda.empty_cache()
        # Update iterator
        idx_ += bs

    model.module.out_features = False
    return feats_query_batched


def estimate_target(args):
    dataset = args.data_dicts_train

    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ]
    )

    sizes = []
    for iSample in dataset:
        # Load image and labels and preprocess
        dict_data = transforms(iSample)
        # Remove any additional element
        label = extract_topk_largest_candidates(np.squeeze(dict_data['label'] == 1).astype(int), 1)

        # Get target size
        sizes.append(label.sum())

    sizes = np.array(sizes)
    if len(sizes) >= 5:
        # Get median size
        median_size = np.median(sizes)
        # Remove outliers
        idx = np.argwhere((np.array(sizes) > (median_size-np.std(sizes))).astype(int) *
                          (np.array(sizes) < (median_size+np.std(sizes))).astype(int))
        # Estimate target
        target_size = np.mean(sizes[idx])
    else:
        target_size = np.mean(sizes)

    return target_size