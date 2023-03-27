import argparse
import os
import numpy as np
import torch
import json
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from fseft.models.utils import load_foundational_model, set_model_finetuning
from fseft.datasets.dataset import get_loader
from fseft.transductive.TI_modules import ProportionConstraint
from utils.losses import BinaryDice3D
from utils.metrics import dice_score, extract_topk_largest_candidates
from utils.misc import set_seeds
from utils.scheduler import LinearWarmupCosineAnnealingLR
from utils.visualization import visualize_prediction
from pretrain.datasets.utils import UNIVERSAL_TEMPLATE

# Check training hardware gpu/cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
# set_seeds(42, use_cuda=device is 'cuda')


def train(args, train_loader, model, optimizer):
    model.train()

    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(device).to(torch.float32), batch["label"].to(device).to(torch.float32)

        # Forward
        logit_map = model(x)

        # Activation
        pred = torch.sigmoid(logit_map)

        # Compute loss
        dsc_loss = BinaryDice3D()(pred, y)

        # Backward and models update
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


def test(args, model):
    model.eval()

    # Prediction
    x = torch.tensor(args.query["image"], dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        yhat = sliding_window_inference(x, (args.roi_x, args.roi_y, args.roi_z), 1, model,
                                        overlap=0.25, mode='gaussian')

    if args.method == 'generalization':
        if args.organ == 'gallbladder':
            organ_template = 'gall'
        elif args.organ == 'kidney_left':
            organ_template = 'lkidney'
        else:
            organ_template = args.organ

        idx = np.argwhere(np.array(list(UNIVERSAL_TEMPLATE.keys())) == organ_template)[0,0]
        yhat = yhat[:, idx, :, :, :]

    if args.postprocess:
        yhat = (yhat > 0.5).squeeze().cpu().detach().numpy()
        yhat = extract_topk_largest_candidates(yhat, 1)
        yhat = torch.tensor(yhat).unsqueeze(0)

    # Metric extraction
    dice, recall, precision = dice_score(yhat.to(device),
                                         torch.tensor(args.query["label"], dtype=torch.float32).to(device))
    print("DICE: " + str(np.round(dice.item(), 4)))

    if not os.path.isdir(args.out_path + '/visualizations/'):
        os.mkdir(args.out_path + '/visualizations/')

    # Prepare experiment name
    path_out = args.out_path + '/visualizations/' + args.method + '_' + args.classifier + '_' + args.organ + '_k_' +\
               str(args.k) + '_fold_' + str(args.iFold)
    if args.transductive_inference:
        path_out = path_out + '_transductive'

    # Save visualization of method
    mask = (yhat.detach().cpu().numpy()) > 0.5
    visualize_prediction(np.squeeze(args.query["image"]), np.squeeze(args.query["label"]), np.squeeze(mask), path_out)

    # Save gt visualization if it does not exist
    path_out = args.out_path + '/visualizations/' + 'gt' + '_' + args.organ + '_fold_' + str(args.iFold)
    if not os.path.isfile(path_out):
        visualize_prediction(np.squeeze(args.query["image"]), np.squeeze(args.query["label"]),
                             np.squeeze(args.query["label"]), path_out)

    return np.round(dice.item(), 4)


def process(args):

    # Set partitions from path and target organ
    args.data_txt_path = {'train': args.data_txt_path + args.organ + '_train.txt',
                          'test': args.data_txt_path + args.organ + '_test.txt'}

    # In the case of fine-tuning with all train subset, we only train once, and test on three folds
    if args.k >= 30:
        args.folds = [1]

    # Set environment for distributed learning
    rank = 0
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # Run training-testing for each fold
    dice_all = []
    for iFold in args.folds:
        args.iFold = iFold
        print(args.method + '_' + args.classifier + '_' + args.organ + '_k_' + str(args.k) +
              '_fold_' + str(args.iFold), end="\n")

        # Set datasets
        train_loader, train_sampler = get_loader(args)

        # Load pretrained models
        model = load_foundational_model(args)

        # Modify base models for fine-tuning
        model = set_model_finetuning(model, args)

        # Set device
        model.to(device)

        # Init prototype-based inferece
        if args.classifier == 'prototype':
            model.classifier[-1].init_prototype(model, train_loader, args.method)
            torch.cuda.empty_cache()

        # Set model to distributed
        if args.dist:
            model = DistributedDataParallel(model, device_ids=[args.device])

        # Set optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch,
                                                      max_epochs=args.epochs,
                                                      warmup_start_lr=args.lr / args.warmup_epoch)

        # If required, init transductive inference module - based on proportion constraints
        if args.transductive_inference:
            args.TI = ProportionConstraint(args, model)

        # Train model
        if not args.method == 'generalization':
            args.epoch = 1
            while args.epoch < args.epochs:
                if args.dist:
                    dist.barrier()
                    train_sampler.set_epoch(args.epoch)

                # Train epoch
                loss_dice = train(args, train_loader, model, optimizer)
                if args.transductive_inference:
                    if args.epoch >= args.pretrain_epochs:
                        TI_penalty = args.TI.step(model, optimizer)
                        torch.cuda.empty_cache()
                        print('Epoch=%d: TI_penalty=%2.5f' % (args.epoch, TI_penalty))

                # Update epoch
                args.epoch += 1

                # Update optimizer scheduler
                if args.scheduler:
                    scheduler.step()

        # Test models
        if args.k >= 30:  # Fine-tuning on all train subset - we only train one model
            for iFold in [1, 2, 3, 4, 5]:
                args.iFold = iFold
                _, _ = get_loader(args)
                dice_fold = test(args, model)
                dice_all.append(dice_fold)
        else:   # Few-shot
            dice_fold = test(args, model)
            dice_all.append(dice_fold)

        if args.dist:
            dist.barrier()

    # Save results and the end of cross-validation
    if rank == 0:
        # Prepare path results and experiment id
        path_results = args.out_path + '/' + args.organ + '_results.json'
        exp_id = args.method + '_' + args.classifier + '_k_' + str(args.k)
        if args.transductive_inference:
            exp_id = exp_id + '_transductive'

        # Save the results
        if not os.path.isfile(path_results):
            d = {exp_id: [dice_all, np.mean(dice_all)]}
        else:
            with open(path_results) as infile:
                d = json.load(infile)
            d[exp_id] = [dice_all, np.mean(dice_all)]
        with open(path_results, 'w') as fp:
            json.dump(d, fp)

    print("Average DICE: " + str(np.mean(dice_all)))

    if args.dist:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    # Folders, dataset, etc.
    parser.add_argument('--data_root_path',
                        default="./data/14_TotalSegmentator/",
                        help='data root path')
    parser.add_argument('--model_path', default='./fseft/pretrained_weights/pretrained_model.pth',
                        help='path of foundational models')
    parser.add_argument('--data_txt_path', default='./fseft/datasets/partitions_main/',
                        help='path with partitions')
    parser.add_argument('--organ', default='kidney_left', help='Target organ')
    parser.add_argument('--folds', default=[1, 2, 3, 4, 5], help='number of testing folds')

    # Storing results
    parser.add_argument('--out_path', default='./fseft/results/')

    # Training options
    parser.add_argument('--k', default=1, type=int, help='Number of shots')
    parser.add_argument('--method', default='linear_probe',
                        help='Options: scratch - generalization - finetuning_all - finetuning_last -'
                             'linear_probe - spatial_adapter ')
    parser.add_argument('--classifier', default='linear', help='linear - prototype')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_samples', default=6, type=int)
    parser.add_argument('--lr', default=0.5, type=float, help='Learning rate')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--transductive_inference', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--margin', default=0.3, type=float, help='margin for proportion constraint')
    parser.add_argument('--lambda_TI', default=0.1, type=float, help='TI relative weigth')
    parser.add_argument('--pretrain_epochs', default=50, type=float, help='Pretrain epochs before TI')
    parser.add_argument('--adapter_kernel_size', default=3, type=int, help='adapter kernel size')

    # Resources
    parser.add_argument('--dist', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument('--num_workers', default=1, type=int, help='workers number for DataLoader')

    # Volume pre-processing
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

    # Predicitons post-processing
    parser.add_argument('--postprocess', default=True, type=lambda x: (str(x).lower() == 'true'))

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()