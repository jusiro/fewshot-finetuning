import torch
import argparse
import os
import torch.distributed as dist

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from pretrain.models.model_pretrain import SwinUNETR
from utils.models import load_weights
from utils.scheduler import LinearWarmupCosineAnnealingLR
from pretrain.datasets.dataset_pretrain import get_loader
from pretrain.datasets.utils import UNIVERSAL_TEMPLATE
from utils.losses import BinaryDice3D
from utils.misc import set_seeds

# Check training hardware gpu/cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
set_seeds(42, use_cuda=device is 'cuda')

NUM_CLASS = len(UNIVERSAL_TEMPLATE.keys())


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
        dsc_loss = BinaryDice3D()(pred, y, annotation_mask=batch['annotation_mask'].to(device).to(torch.float32))

        # Backward and model update
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


def process(args):
    args.NUM_CLASS = NUM_CLASS
    # Set enviroment for distributed learning
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # Set model
    model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z), in_channels=1, out_channels=NUM_CLASS,
                      feature_size=48, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False)

    # Load pretrained weights from encoder and set device
    model = load_weights(model, args.pretrained_model).train().to(device)

    if args.resume:
        model = load_weights(model, args.resume_model_id, classifier=True)

    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])

    # Set optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch,
                                              warmup_start_lr=args.lr/args.warmup_epoch)

    # Set datasets
    train_loader, train_sampler = get_loader(args)

    # Train model
    args.epoch = args.last_epoch
    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)

        # Train epoch
        loss_dice = train(args, train_loader, model, optimizer)

        # Save model
        if (args.epoch % args.store_num == 0) and rank == 0:

            if not os.path.isdir(args.out_path):
                os.mkdir(args.out_path)
            torch.save(model.state_dict(),
                       args.out_path + '/' + 'pretrained_epoch' + str(args.epoch) + '.pth')
            print('save model success')

        # Update epoch
        args.epoch += 1
        # Update optimizer scheduler
        scheduler.step()

    if args.dist:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    # Folders, dataset, etc.
    parser.add_argument('--out_path', default='./pretrain/results/', help='The path resume from checkpoint')
    parser.add_argument('--data_root_path', default="./data/", help='data root path')
    parser.add_argument('--stage', default="train", help='train/val')
    parser.add_argument('--data_txt_path', default={'train': './pretrain/datasets/partial.txt'}, help='data txt path')
    parser.add_argument('--partitions', default=['train'], help='partitions to include in the dataset')

    # Training options
    parser.add_argument('--max_epoch', default=800, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=5 * 1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--classifier', default='linear', help='type of classifier')
    parser.add_argument('--text_controller_type', default='word', help='type of text controller')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--balanced', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--shuffle', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--pretrained_model', default='./pretrain/pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt',
                        help='The path of pretrain model')

    # Resources
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
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

    # Resume training options
    parser.add_argument('--resume', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--resume_model_id', default=None)
    parser.add_argument('--last_epoch', default=1, type=int)

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()