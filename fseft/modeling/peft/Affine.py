

def set_peft(model, args):
    freeze_params(model, args)


def set_training_mode(model, args):
    model.eval()

    if args.model_cfg["architecture"] == "unet3d":
        # Set only encoder weights to trainable
        for name, layer in model.named_children():
            if "down" in name:
                layer.train()


def set_bb(args):
    return []


def freeze_params(model, args):
    print("Freezing weights... ", end="\n")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    print("Unfreeze batch/instance norm weights... ", end="\n")
    for name, param in model.named_parameters():
        # For 3DUnet encoder
        if "down_tr" in name:
            if "bn1" in name:
                param.requires_grad = True
        # For swinvit encoder
        if 'swinViT' in name:
            if 'norm' in name:
                param.requires_grad = True

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