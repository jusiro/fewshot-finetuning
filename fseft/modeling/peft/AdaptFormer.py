# The Adapter codes are adapted from (https://github.com/ShoufaChen/AdaptFormer)
import math
import torch
import torch.nn as nn


def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)
    # Modify encoder with Compacter
    model.swinViT = SwinViTWrapper(model.swinViT)


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
            if args.model_id == "selfsup":
                if "encoder" in name:  # swin-unetr bottleneck
                    param.requires_grad = True


class SwinTransformerBlockWrapper(torch.nn.Module):
    def __init__(self, block, rank=16):
        super().__init__()

        # Store original block
        self.SwinTransformerBlock = block
        self.rank = rank
        self.dim = self.SwinTransformerBlock.dim

        # Insert adapters
        self.mlpAdapter = MLPAdapter(self.dim, self.dim, rank=self.rank)

    def forward(self, x, mask_matrix):
        shortcut = x
        # Forward multi-head attention
        x = self.SwinTransformerBlock.forward_part1(x, mask_matrix)
        # Residual part 1
        x = shortcut + self.SwinTransformerBlock.drop_path(x)

        shortcut = x
        # Forward feedforward + MLPAdapter
        x_mlp = self.SwinTransformerBlock.forward_part2(x)
        x = self.mlpAdapter(x, x_mlp)
        # Residual part 2
        x = shortcut + x

        return x


class SwinViTWrapper(torch.nn.Module):
    def __init__(self, vit_model, rank=4):
        super(SwinViTWrapper, self).__init__()
        self.ViTbase = vit_model
        self.rank = rank

        for i, layer in enumerate(list(self.ViTbase.children())[1:]):
            for ii, blocks in enumerate(layer.children()):
                for iii, block in enumerate(list(blocks.children())[0]):
                    list(blocks.children())[0][iii] = SwinTransformerBlockWrapper(block)

    def forward(self, x, mask_matrix):
        return self.ViTbase(x, mask_matrix)


class MLPAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=16, scale=0.1):
        super().__init__()

        # Define hyperparams
        self.scale = float(scale)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        # Define layers
        self.down_proj = nn.Linear(self.in_features, self.rank)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.out_features)

        # Init new layers from adapter
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, x_mlp):

        # Adapter branch
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        up = self.up_proj(down)

        # Scaling
        up = up * self.scale

        # Residual connection
        output = x_mlp + up

        return output