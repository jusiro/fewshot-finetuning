# The Adapter codes are adapted from (https://github.com/srebuffi)
import torch
import torch.nn as nn


def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)
    # Modify encoder with Compacter
    UNet3DWrapper(model)


def set_bb(args):
    return []


def set_training_mode(model, args):
    model.eval()


def freeze_params(model, args):
    print("Freezing weights... ", end="\n")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    # Freeze/unfreeze decoder
    if args.adapt_hp["decoder"] != "frozen":
        print("UnFreezing decoder weights... ", end="\n")
        for name, param in model.named_parameters():
            if "decoder" in name:  # swin-unetr code for decoder
                param.requires_grad = True
            if "up_tr" in name:  # 3d unet code for decoder
                param.requires_grad = True


class DownTransitionWrapper(nn.Module):
    def __init__(self, block):
        super(DownTransitionWrapper, self).__init__()
        self.block = block
        self.Adapter1 = ParallelAdapter(self.block.in_channel, self.block.out_chan // 2, block.ops[0])
        self.Adapter2 = ParallelAdapter(self.block.out_chan // 2, self.block.out_chan, block.ops[1])

    def forward(self, x):
        if self.block.current_depth == 3:
            out = self.Adapter2(self.Adapter1(x))
            out_before_pool = out
        else:
            out_before_pool = self.Adapter2(self.Adapter1(x))
            out = self.block.maxpool(out_before_pool)
        return out, out_before_pool


class UNet3DWrapper(torch.nn.Module):
    def __init__(self, UNet3D):
        super(UNet3DWrapper, self).__init__()
        self.UNet3D = UNet3D

        for i, (name, block) in enumerate(self.UNet3D.named_children()):
            if "down" in name:
                setattr(self.UNet3D, name, DownTransitionWrapper(block))

    def forward(self, x):
        return self.UNet3D(x)


class ParallelAdapter(nn.Module):
    def __init__(self, in_features, out_features, orig_layers):
        super().__init__()

        # Define layers
        self.orig_layers = orig_layers
        self.alpha = nn.Conv3d(in_features, out_features, kernel_size=1, padding="same", bias=False)

    def forward(self, x):
        # Conv + Residual Adapter
        x = self.orig_layers.conv1(x) + self.alpha(x)
        # bn
        x = self.orig_layers.bn1(x)
        # act
        output = self.orig_layers.activation(x)
        return output