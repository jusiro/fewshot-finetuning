# Few-Shot Efficient Fine-Tuning

With the recent raise of foundation models in computer vision and NLP, the *pretrain-and-adapt* strategy, where a large-scale model is fine-tuned on downstream tasks, is gaining popularity.
However, traditional fine-tuning approaches may still require significant resources and yield sub-optimal results when the labeled data of the target task is scarce.
This is especially the case in clinical settings. To address this challenge, we formalize **few-shot efficient fine-tuning (FSEFT)**, a novel and realistic setting for medical image segmentation.

This repository contains a framework for building novel adapters and few-shots learning strategies for fine-tuning foundational models in medical image segmentation of volumetric data.

<img src="./documents/overview.png" width = "750" alt="" align=center /> <br/>

<b>Transductive few-shot adapters for medical image segmentation</b> <br/>
[Julio Silva-Rodriguez](https://scholar.google.es/citations?user=1UMYgHMAAAAJ&hl), [Jose Dolz](https://scholar.google.es/citations?user=yHQIFFMAAAAJ&hl),
[Ismail Ben Ayed](https://scholar.google.es/citations?user=29vyUccAAAAJ&hl) <br/>
[LIVIA](https://liviamtl.ca/), ETS, Montreal <br/>
[paper](https://arxiv.org/abs/2303.17051) | [code](https://github.com/jusiro/fewshot-finetuning) |

## 💡 Preparation
Create enviroment, clone repository and install required packages (check compatibility with your cuda version).

```
conda create -n fseft python=3.8 -y
conda activate fseft
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c menpo wget
pip install -r requirements.txt
cd ./pretrain/pretrained_weights/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
cd ../
```

## 📦 Foundation model pretraining

The foundational model is trained on CT scans, using 29 different anatomical structures (see `pretrain/datasets/utils.py` for details), and 9 different datasets:
[BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789),
[CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/),
[LiTS](https://competitions.codalab.org/competitions/17094#learn_the_details),
[KiTS](https://kits21.kits-challenge.org/participate#download-block),
[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K),
[AMOS](https://amos22.grand-challenge.org),
[MSD](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2),
[AcdomenCT-12organ](https://github.com/JunMa11/AbdomenCT-1K) 
and [CT-org](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890).
The total 2022 CT scans used for training are indicated in `pretrain/datasets/train.txt`. Foundational model is trained using 4 NVIDIA RTX A600 GPUs, using distributed learning as follows:

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_pretrain.py --dist True --num_workers 6 --batch_size 2 --num_samples 3 --max_epoch 1000 --lr 1e-4 --balanced True
```

The pretrained model is accesible [here](https://drive.google.com/file/d/18yLNxmWGnVifQNeYYwyyu56Cg4tWV9aW/view?usp=sharing). For usage, please download it and place it in: `fseft/pretrained_weights/pretrained_model.pth`

## 📦 Transductive few-shot adapters

The experiments for adaptation of foundational model are performed on [TotalSeg](https://zenodo.org/record/6802614#.ZBDA3dLMKV4) dataset, using the scans of one unique study type and center. 

You can train the few-shot spatial adapter module as follows:

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=2,3 python -W ignore -m torch.distributed.launch --nproc_per_node=2 --master_port=1235 main_fseft.py --dist True --num_workers 4 --batch_size 1 --num_samples 6 --k 5 --lr 1e-2 --organ esophagus --epochs 100 --method spatial_adapter --classifier linear --transductive False --scheduler True
```

By using additional resources at test-time, the inference can be performed in a transductive fashion, taking into account anatomical proportion constraints:

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=0,2 python -W ignore -m torch.distributed.launch --nproc_per_node=2 --master_port=1235 main_fseft.py --dist True --num_workers 4 --batch_size 1 --num_samples 6 --k 10 --lr 1e-2 --organ esophagus --epochs 100 --method spatial_adapter --classifier linear --scheduler True --transductive_inference True --pretrain_epochs 50 --margin 0.3 --lambda_TI 0.1```
```

### **How to perform adaptation to new domains/tasks?**

1. Adequate dataset: use `.nii` volumes, and binary masks as annotations volumes (i.e. `0:background | 1: target`).  
2. Prepare the partition files, named `[ORGAN]_TRAIN.txt` and `[ORGAN]_test.txt`. The files must present 2 columns, for volume and annotations path, similarly to `./fseft/datasets/partitions_main/liver_train.txt`
3. Locate yor foundational model on `./fseft/pretrained_weights/` or use the proposed pretrained model.
4. Modify input flags for `main_fseft.py`, i.e. `--data_root_path, --model_path, --data_txt_path, etc.`,  and train the adapter module.

## 🙏 Acknowledgement

The framework is build upon the [monai](https://github.com/Project-MONAI/MONAI) library for medical image segmentation.

The implementation and pre-training of the foundational model is largely based on the [UniversalModel](https://github.com/ljwztc/CLIP-Driven-Universal-Model) implementation.


## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@inproceedings{FSEFT,
  title={Towards Foundation Models and Few-Shot Parameter-Efficient Fine-Tuning for Volumetric Organ Segmentation},
  author={Julio Silva-Rodríguez and Jose Dolz and Ismail {Ben Ayed}},
  booktitle={Int. Workshop on Foundation Models for General Medical AI (MICCAIw)},
  year={2023}
}
```
