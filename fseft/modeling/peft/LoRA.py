import torch


def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)
    # Modify encoder with LoRA
    model.swinViT = SwinViTLoRAWrapper(model.swinViT)


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


class LoRALayer(torch.nn.Module):
    def __init__(self, w, w_a, w_b):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class SwinViTLoRAWrapper(torch.nn.Module):
    def __init__(self, vit_model, rank=4):
        super(SwinViTLoRAWrapper, self).__init__()
        self.ViTbase = vit_model
        self.rank = rank

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        for i, layer in enumerate(list(self.ViTbase.children())[1:]):
            for ii, blocks in enumerate(layer.children()):
                for iii, block in enumerate(list(blocks.children())[0]):
                    w_a_linear_qkv = torch.nn.Linear(block.attn.qkv.in_features, self.rank*3, bias=False)
                    w_b_linear_qkv = torch.nn.Linear(self.rank*3, block.attn.qkv.out_features, bias=False)
                    torch.nn.init.zeros_(w_b_linear_qkv.weight)
                    self.w_As.append(w_a_linear_qkv)
                    self.w_Bs.append(w_b_linear_qkv)
                    block.attn.qkv = LoRALayer(block.attn.qkv, w_a_linear_qkv, w_b_linear_qkv)

    def forward(self, x, mask_matrix):
        return self.ViTbase(x, mask_matrix)
