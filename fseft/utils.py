import torch

from models.architectures.SwinUNETR import SwinUNETR
from models.architectures.unet3d import UNet3D
from utils.models import load_weights

from fseft.modeling.heads import LinearProbe, retrieve_zeroshot_weights
from fseft.configs import get_adapt_config


def set_model_peft(model, args):

    # Retrieve training setting
    get_adapt_config(args)

    # 1. Set backbone for PEFT (freeze/unfreeze + add parameters)
    args.adapt_hp["set_peft"](model, args)

    # 2. Set black-box adaptation
    modules = args.adapt_hp["set_bb"](args)

    # 3. Set classification head
    modules.append(LinearProbe(feature_size=args.model_cfg["fout"], out_channels=args.out_channels,
                               zs_weights=retrieve_zeroshot_weights(args, model)))
    model.classifier = torch.nn.Sequential(*modules)

    # Print trainable parameters:
    print_train_parameters(model)

    return model


def load_model(args):
    # Set model
    if args.model_cfg["architecture"] == "swinunetr":

        model = SwinUNETR(img_size=(args.model_cfg["roi_x"], args.model_cfg["roi_y"], args.model_cfg["roi_z"]),
                          out_channels=args.model_cfg["channelOut"], feature_size=args.model_cfg["fout"],
                          in_channels=1, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False)

    elif args.model_cfg["architecture"]:
        model = UNet3D()
    else:
        print("WARNING: architecture not supported!")
        model = UNet3D()

    # Load weights
    if args.method == "scratch":
        args.model_cfg["model_path"] = "./models/pretrained_weights/selfsup.pt"
    model = load_weights(model, args.model_cfg["model_path"], classifier=(args.method != 'scratch'),
                         decoder=args.decoder, bottleneck=args.bottleneck)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_train_parameters(model):
    # Print trainable parameters:
    print("Number of trainable parameters: " + str(count_parameters(model)))
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(name + " " * (70 - len(name)) + " -> Trained:" + str(param.requires_grad))

