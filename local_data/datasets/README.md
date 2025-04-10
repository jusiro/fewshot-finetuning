We suggest the following dataset organization to ease management and avoid modifying the source code.
The datasets structure (for the adaptation experiments) looks like:

```
fewshot-finetuning/
└── local_data/
    └── datasets/
        └── CT/
            ├── 14_TotalSegmentator
            └── 17_FLARE22
```

In the following, we provide specific download links and expected structure for each individual dataset.

### TotalSegmentator

```
.
└── 14_TotalSegmentator/
    └── data/
        ├── s0000/
        │   ├── ct.nii.gz
        │   └── segmentations/
        │       ├── adrenal_gland_left.nii.gz
        │       ├── adrenal_gland_right.nii.gz
        │       ├── aorta.nii.gz
        │       ├── autochthon_left.nii.gz
        │       ├── autochthon_right.nii.gz
        │       ├── brain.nii.gz
        │       └── ...
        ├── s0001/
        │   └── ...
        ├── s0002/
        │   └── ...
        ├── s0003/
        │   └── ...
        └── ...
```

### FLARE'22

```
.
└── 17_FLARE22/
    ├── train/
    │   ├── images/
    │   │   ├── FLARE22_Tr_0001_0000.nii.gz
    │   │   ├── FLARE22_Tr_0002_0000.nii.gz
    │   │   └── ...
    │   └── labels/
    │       ├── FLARE22_Tr_0001.nii.gz
    │       ├── FLARE22_Tr_0002.nii.gz
    │       └── ...  
    └── tuning/
        ├── images/
        │   ├── FLARETs_0001_0000.nii.gz
        │   ├── FLARETs_0002_0000.nii.gz
        │   └── ...
        └── labels/
            ├── FLARETs_0001.nii.gz
            ├── FLARETs_0001.nii.gz
            └── ...
```
