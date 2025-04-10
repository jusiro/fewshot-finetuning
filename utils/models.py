import torch


def load_weights(model, weights_path, classifier=False, decoder="fine-tuned", bottleneck="fine-tuned"):
    print('Use pretrained weights')

    model_dict = torch.load(weights_path)
    if 'state_dict' in list(model_dict.keys()):
        state_dict = model_dict["state_dict"]
    elif 'net' in list(model_dict.keys()):
        state_dict = model_dict["net"]
    else:
        state_dict = model_dict

    # fix potential differences in state dict keys from pre-training to fine-tuning
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "backbone." in list(state_dict.keys())[10]:
        print("Tag 'backbone' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("backbone.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    # fix potential differences in state dict keys from pre-training to fine-tuning for publicly available modeling
    # For adapting modeling from (https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
    if "out.conv.conv" in list(state_dict.keys())[-1]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("out.conv.conv", "classifier")] = state_dict.pop(key)

    if "model_swinvit" in weights_path:
        for key in list(state_dict.keys()):
            state_dict[key.replace(key, "swinViT." + key)] = state_dict.pop(key)
        for key in list(state_dict.keys()):
            if "mlp.fc" in key:
                state_dict[key.replace("mlp.fc", "mlp.linear")] = state_dict.pop(key)

    store_dict = {}
    for key in state_dict.keys():
        if 'classifier' not in key:
            store_dict[key] = state_dict[key]
        else:
            if classifier and 'classifier' in key:
                store_dict[key] = state_dict[key]

    # Check decoder transferability
    if decoder == "new":
        print("Replacing decoder pre-trained weights by a new decoder")
        for key in list(store_dict.keys()):
            if 'decoder' in key or "classifier" in key:
                store_dict.pop(key, None)

    # Check bottleneck layers
    if bottleneck == "new":
        print("Replacing bottleneck pre-trained weights by a new bottleneck")
        for key in list(store_dict.keys()):
            if 'encoder' in key or "classifier" in key:
                store_dict.pop(key, None)

    model.load_state_dict(store_dict, strict=False)

    return model

