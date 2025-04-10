

def get_model_config(args):

    # Ours pre-trained foundation model
    if args.model_id == "fseft":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/fseft.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # SUPrem UNET pre-trained
    elif args.model_id == "suprem_unet":
        model_cfg = {"architecture": "unet3d", "model_path": "./models/pretrained_weights/suprem_unet.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 64, "channelOut": 29}

    # SUPrem UNET pre-trained
    elif args.model_id == "suprem_swinunetr":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/suprem_swinunetr.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # CLIP-driven foundation model
    elif args.model_id == "clipdriven":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/clipdriven.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # Tang et al. (2022) pre-trained on BTCV
    elif args.model_id == "selfsup":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/selfsup.pt",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 14}

    # Tang et al. (2022) pre-trained on BTCV
    elif args.model_id == "btcv":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/btcv.pt",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 14}

    # Parse keys into argparser
    args.model_cfg = model_cfg
