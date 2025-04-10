import numpy as np
import torch

from monai.data import MetaTensor, Dataset
from monai.transforms import (apply_transform, RandZoomd, ScaleIntensityRangePercentilesd,
                              ScaleIntensityRanged)

from pretrain.datasets.templates import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = []
        for img in data:
            keys.append(img["name"].split("/")[0])

        self.datasetkey = list(np.unique(keys))

        data_dic = {}
        for iKey in self.datasetkey:
            data_dic[iKey] = [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
        self.data_dic = data_dic

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]

        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class SelectRelevantKeys():
    def __call__(self, data):
        #if np.array(data["label"]).sum() == 0:
        #    print(data["name"])
        d = {key: data[key] for key in ['image', 'label', 'name']}
        return d


class LRDivision():
    def __call__(self, data):

        if 'KiTS' in data["name"]:
            center_orig = np.array(data["image"].shape)//2

            # rkidney --> lkidney
            mask = data["label"][:, :center_orig[1], :, :] == 1
            data["label"][:, :center_orig[1], :, :][mask] = 51

        if 'AbdomenCT-1K' in data["name"]:
            center_orig = np.array(data["image"].shape)//2

            # rkidney --> lkidney
            mask = data["label"][:, :center_orig[1], :, :] == 2
            data["label"][:, :center_orig[1], :, :][mask] = 52

        if 'CT-ORG' in data["name"]:
            center_orig = np.array(data["image"].shape)//2

            # rlung --> llung
            mask = data["label"][:, :center_orig[1], :, :] == 3
            data["label"][:, :center_orig[1], :, :][mask] = 53
            # rkidney --> lkidney
            mask = data["label"][:, :center_orig[1], :, :] == 4
            data["label"][:, :center_orig[1], :, :][mask] = 54

        if 'AbdomenCT-12organ' in data["name"]:
            center_orig = np.array(data["image"].shape)//2

            # rkidney --> lkidney
            mask = data["label"][:, :center_orig[1], :, :] == 2
            data["label"][:, :center_orig[1], :, :][mask] = 52

        return data


class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = name.split("/")[0]
        if not ("10" in key):
            return d
        d = super().__call__(d)
        return d


class MapLabels():
    def __call__(self, data):
        y = data['label']
        name = data['name']

        # 01. BTCV (Multi-atlas)
        if 'Multi-Atlas' in name:
            template = BTCV_TEMPLATE
        # 03. CHAOS
        elif 'CHAOS' in name:
            template = CHAOS_TEMPLATE
        # 04_LiTS
        elif 'LiTS' in name:
            template = LiTS_TEMPLATE
        # 05_KiTS
        elif 'KiTS' in name:
            template = KiTS_TEMPLATE
        # 08_AbdomenCT-1K
        elif 'AbdomenCT-1K' in name:
            template = AbdomenCT1K_TEMPLATE
        # 09_AMOS
        elif 'AMOS' in name:
            template = AMOS_TEMPLATE
        # 10_Task03_Liver
        elif 'Task03' in name:
            template = DEC_Task03_Liver_TEMPLATE
        # 10_Task06_Lung
        elif 'Task06' in name:
            template = DEC_Task06_Lung_TEMPLATE
        # 10_Task07_Pancreas
        elif 'Task07' in name:
            template = DEC_Task07_Pancreas_TEMPLATE
        # 10_Task08_HepaticVessel
        elif 'Task08' in name:
            template = DEC_Task08_HepaticVessel_TEMPLATE
        # 10_Task09_Spleen
        elif 'Task09' in name:
            template = DEC_Task09_Spleen_TEMPLATE
        # 10_Task10_Colon
        elif 'Task10' in name:
            template = DEC_Task10_Colon_TEMPLATE
        # 12_CT-ORG
        elif 'CT-ORG' in name:
            template = CTORG_TEMPLATE
        # 13_AbdomenCT-12organ
        elif 'AbdomenCT-12organ' in name:
            template = AbdomenCT12_TEMPLATE
        else:
            print("WARNING: Not template found for " + name, end="\r")
            data['annotation_mask'] = np.array(1)
            return data

        try:
            map_dict = {}
            for iKey in template.keys():
                map_dict[template[iKey]] = UNIVERSAL_TEMPLATE[iKey]
            map_dict[0] = 0
            y = np.array(np.vectorize(map_dict.get)(y))

            annotation_mask = np.array([iKey in template for iKey in UNIVERSAL_TEMPLATE.keys()]).astype(int)
        except:
            print("WARNING: Error during mapping. Check that all organs are at the universal tamplate.", end='\n')

        data['annotation_mask'] = annotation_mask
        data['label'] = y
        return data


class CategoricalToOneHot():
    def __init__(self, args):
        self.classes = len(args.dataset_indexes) + 1
        self.objective = args.objective

    def __call__(self, data):
        y = data['label']

        # one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.long), num_classes=self.classes).permute(
            (-1, 1, 2, 3, 0)).squeeze(-1)

        # quit background class
        if self.objective == "binary":
            y = y[1:, :, :, :]

        data['label'] = y
        return data


def select_intensity_scaling(args):
    if "kipa" in args.data_txt_path["train"]:
        transform = ScaleIntensityRangePercentilesd(keys=["image"], lower=10, upper=90, b_min=args.model_cfg["b_min"],
                                                    b_max=args.model_cfg["b_max"], clip=True)
    else:
        transform = ScaleIntensityRanged(keys=["image"], a_min=args.model_cfg["a_min"], a_max=args.model_cfg["a_max"],
                                         b_min=args.model_cfg["b_min"], b_max=args.model_cfg["b_max"], clip=True)
    return transform


class SelectAdaptationOrgan():

    def __init__(self, args):
        self.indexes = args.dataset_indexes

    def __call__(self, data):
        if "KiPA" in data["name"]:  # Select kidney
            data["label"] = MetaTensor(torch.tensor(np.int8(data["label"] == 2)), meta=data["label"].meta)
        if "FLARE" in data["name"]:
            # Select only target categories in mask
            mask = np.int8(np.zeros_like(data["label"]))
            for i in range(len(self.indexes)):
                mask += (np.int8(data["label"] == self.indexes[i]) * (i+1))
            # Create new label mask
            data["label"] = MetaTensor(torch.tensor(mask), meta=data["label"].meta)

        return data