import torch

from pretrain.models.model_pretrain import SwinUNETR
from pretrain.datasets.utils import UNIVERSAL_TEMPLATE
from fseft.models.adapters import SpatialAdapter, LinearProbe, Prototype
from utils.models import load_weights


def set_model_finetuning(model, args):

    # For direct inference, keep model the same
    if args.method == 'generalization':
        return model

    # Freeze layers depending on configuration
    if args.method != 'finetuning_all' and args.method != 'scratch':
        print("Freezing weights... ", end="\n")
        for name, param in model.named_parameters():
            if not 'classifier' in name:
                param.requires_grad = False
    if args.method == 'finetuning_last':
        print("Unfreeze last conv block weights... ", end="\n")
        for name, param in model.named_parameters():
            if 'decoder1' in name:
                param.requires_grad = True

    # Prepare new modules for tuning
    modules = []
    # Set backbone head
    if args.method == 'spatial_adapter':
        modules.append(SpatialAdapter(adapter_kernel_size=args.adapter_kernel_size))
    # Set new classification head
    if args.classifier == 'linear':
        modules.append(LinearProbe())
    elif args.classifier == 'prototype':
        modules.append(Prototype())

    # Set new head
    model.classifier = torch.nn.Sequential(*modules)

    return model


def load_foundational_model(args):
    # Set model
    model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z), in_channels=1,
                      out_channels=len(UNIVERSAL_TEMPLATE), feature_size=48, drop_rate=0.0, attn_drop_rate=0.0,
                      dropout_path_rate=0.0, use_checkpoint=False)

    # Load weights
    if args.method == 'generalization':
        model = load_weights(model, args.model_path, classifier=True)
    elif args.method == 'scratch':
        model = load_weights(model, './pretrain/pretrained_weights/model_swinvit.pt')
    else:
        model = load_weights(model, args.model_path, classifier=False)

    return model


