import argparse
import os
import numpy as np
import torch
import json
import time

from monai.inferers import sliding_window_inference
from tqdm import tqdm

from fseft.utils import load_model, set_model_peft
from models.configs import get_model_config
from fseft.datasets.dataset import get_loader
from fseft.datasets.configs import get_task_config
from fseft.datasets.utils import load_query
from utils.losses import BinaryDice3D
from utils.metrics import dice_score, extract_topk_largest_candidates
from utils.misc import set_seeds
from utils.scheduler import LinearWarmupCosineAnnealingLR
from utils.visualization import visualize_query

# Check training hardware gpu/cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
set_seeds(42, use_cuda=device == 'cuda')


def train(args, train_loader, model, optimizer):
    args.adapt_hp["set_training_mode"](model, args)

    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(device).to(torch.float32), batch["label"].to(device).to(torch.float32)

        # Forward
        logit_map = model(x)

        # Activation
        if args.objective == "binary":
            pred = torch.sigmoid(logit_map)
        elif args.objective == "multiclass":
            pred = torch.softmax(logit_map, dim=1)

        # Compute loss
        dsc_loss = BinaryDice3D()(pred, y)

        # Backward and modeling update
        loss = dsc_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display training track
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f)" % (
                args.epoch, step + 1, len(train_loader), dsc_loss.item())
        )

        # Overall losses track
        loss_dice_ave += dsc_loss.item()
        torch.cuda.empty_cache()

    # Display epoch-wise loss
    print('Epoch=%d: ave_dice_loss=%2.5f' % (args.epoch, loss_dice_ave / len(epoch_iterator)))

    return loss_dice_ave / len(epoch_iterator)


def test(args, model, overlap_inference=0.25):
    model.eval()

    # Retrieve testing image
    x = args.query["image"].detach().clone().to(device).unsqueeze(0)

    # Predict logits
    with torch.no_grad():
        yhat = sliding_window_inference(x, (args.model_cfg["roi_x"], args.model_cfg["roi_y"], args.model_cfg["roi_z"]),
                                        1, model, overlap=overlap_inference, mode='gaussian')

    # Produce categorical predictions
    if args.objective == "binary":
        yhat = (torch.sigmoid(yhat) > 0.5).squeeze(dim=1).cpu().detach().numpy()
    elif args.objective == "multiclass":
        yhat = torch.argmax(yhat, 1).cpu().detach().numpy()

    # Produce class-wise metrics
    dice_all = {}
    for i_organ in range(len(args.selected_organs)):
        # Retrieve pred and mask per class
        pred_i = (yhat == (i_organ+1))
        mask_i = (args.query["label"] == (i_organ+1))

        # Post-process mask
        if args.post_process:
            pred_i = torch.tensor(extract_topk_largest_candidates(pred_i, 1)).unsqueeze(0)

        # Metric extraction
        dice, recall, precision = dice_score(torch.tensor(pred_i, dtype=torch.float32).to(device),
                                             torch.tensor(mask_i, dtype=torch.float32).to(device))
        dice_all[args.selected_organs[i_organ]] = np.round(dice.item(), 4)

    # Store prediction for further visualizations
    args.query["prediction"] = yhat

    return dice_all


def process(args):

    # Get dataset config and partitions
    get_task_config(args)

    # Run adaptation-testing for each fold
    dice_all = []
    for iFold in args.folds:
        t1 = time.time()

        args.iFold = iFold
        print(args.method + '_' + '_' + args.organ + '_k_' + str(args.k) +
              '_fold_' + str(args.iFold), end="\n")

        # Get model config
        get_model_config(args)

        # Init train learning curve
        args.lc = {"loss_dice_train": [], "loss_dice_val": [], "best_epoch": 0, "best_val_dice": 1.0}

        # Load pretrained modeling
        model = load_model(args)

        # Modify base modeling for fine-tuning
        model = set_model_peft(model, args)

        # Set device
        model.to(device)

        # Set datasets
        train_loader = get_loader(args)

        # Set optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adapt_hp["lr"], weight_decay=0.0)
        if args.adapt_hp["scheduler"]:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.adapt_hp["warmup_epoch"],
                                                      max_epochs=args.adapt_hp["epochs"],
                                                      warmup_start_lr=args.adapt_hp["lr"]/args.adapt_hp["warmup_epoch"])

        # Train model
        if not args.method == 'generalization':
            args.epoch, early_stop = 1, False

            while (args.epoch < args.adapt_hp["epochs"]) and (early_stop is False):

                # Train epoch
                loss_dice = train(args, train_loader, model, optimizer)
                args.lc["loss_dice_train"].append(loss_dice)

                # Evaluate if early stopping is set
                if args.early_stop_criteria == "val":
                    model.eval()
                    with torch.no_grad():
                        # Forward
                        logit_map = model(args.data_val_dict["image"].clone().to(device).to(torch.float32))
                        # Activation
                        pred = torch.sigmoid(logit_map)
                        # Compute loss
                        dsc_loss_val = BinaryDice3D()(pred,
                                                      args.data_val_dict["label"].clone().to(device).to(torch.float32))
                        dsc_loss_val = np.round(dsc_loss_val.item(), decimals=4)

                        args.lc["loss_dice_val"].append(dsc_loss_val)
                        print("val loss: " + str(dsc_loss_val))

                # Update epoch
                args.epoch += 1

                # Update optimizer scheduler
                if args.adapt_hp["scheduler"]:
                    scheduler.step()

                # Early stopping based on train convergence
                if args.early_stop_criteria == "train":
                    if args.epoch > (args.adapt_hp["pat"]+11):
                        lc_smooth = list(np.convolve(args.lc["loss_dice_" + args.early_stop_criteria],
                                                     np.ones(10)/10, mode='valid'))
                        if (lc_smooth[-args.adapt_hp["pat"]] - lc_smooth[-1]) < 0.001:
                            print("Train loss on plateau... early stopping")
                            print("Plateau DSC loss: {loss_now} -- Old DSC loss: {loss_old}".format(
                                loss_now=str(round(lc_smooth[-1], 4)),
                                loss_old=str(round(lc_smooth[-args.adapt_hp["pat"]], 4))))
                            early_stop = True
                # Early stopping based on validation convergence
                elif args.early_stop_criteria == "val":
                    if dsc_loss_val < args.lc["best_val_dice"]:
                        print("Validation loss improved...")
                        args.lc["best_val_dice"] = dsc_loss_val
                        args.best_weights = model.state_dict()
                        c = 1
                    else:
                        c += 1
                        if c > args.adapt_hp["patience"]:
                            print("Val loss on plateau... early stopping - loading best model")
                            early_stop = True
                            model.load_state_dict(args.best_weights)

        t2 = time.time()
        minutes, seconds = divmod(t2 - t1, 60)
        hours, minutes = divmod(minutes, 60)
        print("Training time: %d:%02d:%02d" % (hours, minutes, seconds))

        # Testing the adaptation stage
        dice_test = []
        for iFold in range(args.ntest):
            print("Testing sample: " + str(iFold + 1))
            args.iFold = iFold
            # Load testing sample
            load_query(args)
            # Evaluate model
            dice_sample = test(args, model)
            dice_test.append(dice_sample)
            # Visualization of results
            if args.visualization:
                visualize_query(args)

        dice_test = dict(zip(
            list(dice_test[0].keys()),
            [np.round(np.mean([isample[organ] for isample in dice_test])*100, 2) for organ in args.selected_organs]
        ))

        print("Average DSC: " + str(dice_test))
        dice_all.append(dice_test)

    # Average across seeds
    dice_all = dict(zip(
        list(dice_all[0].keys()),
        [np.round(np.mean([isample[organ] for isample in dice_all]), 2) for organ in args.selected_organs]
    ))

    # Save results and the end of cross-validation
    path_results = args.out_path + args.organ + '_results.json'
    exp_id = args.method + '_k_' + str(args.k) + "_" + args.decoder + "Decoder"

    # Save the results_prev
    if not os.path.isfile(path_results):
        d = {exp_id: dice_all}
    else:
        with open(path_results) as infile:
            d = json.load(infile)
        d[exp_id] = dice_all
    with open(path_results, 'w') as fp:
        json.dump(d, fp)

    print("Results across folds.")
    print("Average DICE: " + str(dice_all))


def main():
    parser = argparse.ArgumentParser()

    # Folders, dataset, etc.
    parser.add_argument('--out_path', default='./local_data/results/')

    # Experiment: model, dataset, training shots, ...
    parser.add_argument('--model_id', default='fseft',
                        help='Options: fseft - selfsup -btcv - clipdriven - suprem_unet - suprem_swinunetr')
    parser.add_argument('--dataset', default="totalseg", help='Options: totalseg - flare')
    parser.add_argument('--organ', default='duodenum',
                        help='Target organ - e.g.: selected - spleen - kidney_left - gallbladder - esophagus          '
                             '                     liver - pancreas - stomach - duodenum - aorta - heart_myocardium   '
                             '                     heart_atrium_left - heart_atrium_right - heart_ventricle_left      '
                             '                     heart_ventricle_right - lung_lower_lobe_left                       '
                             '                     lung_lower_lobe_right - lung_middle_lobe_right                     '
                             '                     lung_upper_lobe_left - lung_upper_lobe_right                       '
                             '                     gluteus_maximus_left - gluteus_maximus_right - gluteus_medius_left '
                             '                     gluteus_medius_right - gluteus_minimus_left - gluteus_minimus_right'
                        )
    parser.add_argument('--k', default=1, type=int, help='Number of shots')
    parser.add_argument('--seeds', default=3, type=int, help='number of experimental seeds')

    # Training hyper-params
    parser.add_argument('--early_stop_criteria', default='train', help='train - val - none')

    # PEFT method for encoder
    parser.add_argument('--method', default='LP',
                        help='No adaptation:              | generalization                          | '
                             'Full FT:                    | scratch - FT                            | '
                             'PEFT:                                                                 | '
                             '  Selective:                | Bias - Affine                           | '
                             '  Additive(adapter):        | AdapterCNN - Compacter - AdaptFormer    | '
                             '  Additive(reparam.):       | LoRA                                    | '
                             'Black-box (bb) adapters:    | LP - bb3DAdapter - bbAdapter            | '
                        )

    # Transferring bottleneck/decoder
    parser.add_argument('--decoder', default="frozen", help='| frozen | fine-tuned | new |')
    parser.add_argument('--bottleneck', default="frozen", help='| frozen | fine-tuned | new |')

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    # Predictions post_process
    parser.add_argument('--post_process', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Predictions visualization
    parser.add_argument('--visualization', default=False, type=lambda x: (str(x).lower() == 'true'))

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()