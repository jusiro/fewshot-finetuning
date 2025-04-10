import os.path

import numpy as np

from local_data.constants import PATH_PARTITIONS_TRANSFERABILITY
from pretrain.datasets.templates import *


def get_task_config(args):

    # Set output folder
    args.out_path = args.out_path + args.dataset + "/" + args.model_id + "/"
    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    # Dataset partition train and test partition
    if args.dataset == "totalseg":
        args.selected_organs = [args.organ]

        # Data partitions
        args.data_txt_path = PATH_PARTITIONS_TRANSFERABILITY + "totalseg_selected/"
        args.data_txt_path = {'train': args.data_txt_path + args.organ + '_train.txt',
                              'test': args.data_txt_path + args.organ + '_test.txt'}
        # Testing cases
        args.ntest = 5
        # Type of task
        args.objective = "binary"
        # Target category in mask
        args.dataset_indexes = [1]  # Label volumes in total-seg are per class, so only one class annotated per mask.
        args.out_channels = 1

        # Target category in universal template (for aligning with pre-trained model)
        if args.organ in ['gallbladder', 'kidney_left']:
            mapping = {'gallbladder': 'gall', 'kidney_left': 'lkidney'}
            args.universal_indexes = [UNIVERSAL_TEMPLATE[mapping[args.organ]]]
        else:
            if args.organ in list(UNIVERSAL_TEMPLATE.keys()):
                args.universal_indexes = [UNIVERSAL_TEMPLATE[args.organ]]
            else:
                args.universal_indexes = None

    elif args.dataset == "flare":
        # Data partitions
        args.data_txt_path = PATH_PARTITIONS_TRANSFERABILITY + "flare/"
        args.data_txt_path = {'train': args.data_txt_path + '_train.txt',
                              'test': args.data_txt_path + '_test.txt'}
        # Testing cases
        args.ntest = 22

        if args.organ == "selected":
            args.selected_organs = ['spleen', 'lkidney', 'gall', 'esophagus', 'liver', 'pancreas', 'stomach',
                                    'duodenum', 'aorta']
            # Type of task
            args.objective = "multiclass"
            args.out_channels = len(args.selected_organs) + 1
            # Target category in mask
            args.dataset_indexes = [FLARE_TEMPLATE[ikey] for ikey in args.selected_organs]
            args.universal_indexes = [UNIVERSAL_TEMPLATE[ikey] for ikey in args.selected_organs]
        else:
            # Type of task
            args.objective = "binary"
            args.out_channels = 1
            if args.organ in ['gallbladder', 'kidney_left']:
                mapping = {'gallbladder': 'gall', 'kidney_left': 'lkidney'}
                args.selected_organs = [mapping[args.organ]]
                # Target category in mask
                args.dataset_indexes = [FLARE_TEMPLATE[ikey] for ikey in [mapping[args.organ]]]
                # Target category in universal template (for aligning with pre-trained model)
                args.universal_indexes = [UNIVERSAL_TEMPLATE[mapping[args.organ]]]
            else:
                args.selected_organs = [args.organ]
                # Target category in mask
                args.dataset_indexes = [FLARE_TEMPLATE[ikey] for ikey in [args.organ]]
                # Target category in universal template (for aligning with pre-trained model)
                args.universal_indexes = [UNIVERSAL_TEMPLATE[args.organ]]

    # Set list of training seeds (repetitions)
    if args.k >= 30:  # In the case of fine-tuning with all train subset, we only train once.
        args.folds = [1]
    else:
        args.folds = list(np.arange(1, args.seeds + 1))
