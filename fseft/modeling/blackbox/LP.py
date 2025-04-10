

def set_peft(model, args):
    # Freeze params
    freeze_params(model, args)


def set_training_mode(model, args):
    model.eval()


def set_bb(args):
    return []


def freeze_params(model, args):
    print("Freezing weights... ", end="\n")
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze/unfreeze decoder encoder
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