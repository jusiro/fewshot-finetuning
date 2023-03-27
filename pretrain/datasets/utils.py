import numpy as np
import torch

from monai.data import Dataset
from monai.transforms import (apply_transform, RandZoomd)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Unified template
UNIVERSAL_TEMPLATE = {'spleen': 1, 'rkidney': 2, 'lkidney': 3, 'gall': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7,
                      'aorta': 8, 'postcava': 9, 'psv': 10, 'pancreas': 11, 'radrenal': 12, 'ladrenal': 13,
                      'duodenum': 14, 'bladder': 15, 'prost_ut': 16, 'liver_tumor': 17, 'kidney_tumor': 18,
                      'kidney_cyst': 19, 'celiac_truck': 20, 'rlung': 21, 'llung': 22, 'bone': 23, 'brain': 24,
                      'lung_tumor': 25, 'pancreas_tumor': 26, 'hv': 27, 'hvt': 28, 'colon_tumor': 29}
# 01. BTCV (Multi-atlas)
BTCV_TEMPLATE = {'spleen': 1, 'rkidney': 2, 'lkidney': 3, 'gall': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7,
                 'aorta': 8, 'postcava': 9, 'psv': 10, 'pancreas': 11, 'radrenal': 12, 'ladrenal': 13}
# 03_CHAOS
CHAOS_TEMPLATE = {'liver': 1}
# 04_LiTS
LiTS_TEMPLATE = {'liver': 1, 'liver_tumor': 2}
# 05_KiTS
KiTS_TEMPLATE = {'rkidney': 1, 'kidney_tumor': 2, 'kidney_cyst': 3, 'lkidney': 51}
# 08_AbdomenCT-1K
AbdomenCT1K_TEMPLATE = {'liver': 1, 'rkidney': 2, 'spleen': 3, 'pancreas': 4, 'lkidney': 52}
# 09_AMOS
AMOS_TEMPLATE = {'spleen': 1, 'rkidney': 2, 'lkidney': 3, 'gall': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7,
                 'aorta': 8, 'postcava': 9, 'pancreas': 10, 'radrenal': 11, 'ladrenal': 12,
                 'duodenum': 13, 'bladder': 14, 'prost_ut': 15}
# 10_Task03_Liver
DEC_Task03_Liver_TEMPLATE = {'liver': 1, 'liver_tumor': 2}
# 10_Task06_Lung
DEC_Task06_Lung_TEMPLATE = {'lung_tumor': 1}
# 10_Task07_Pancreas
DEC_Task07_Pancreas_TEMPLATE = {'pancreas': 1, 'pancreas_tumor': 2}
# 10_Task08_HepaticVessel
DEC_Task08_HepaticVessel_TEMPLATE = {'hv': 1, 'hvt': 2}
# 10_Task09_Spleen
DEC_Task09_Spleen_TEMPLATE = {'spleen': 1}
# 10_Task10_Colon
DEC_Task10_Colon_TEMPLATE = {'colon_tumor': 1}
# 12_CT-ORG
CTORG_TEMPLATE = {'liver': 1, 'bladder': 2, 'rlung': 3, 'rkidney': 4, 'bone': 5, 'brain': 6,
                  'llung': 53, 'lkidney': 54}
# 13_AbdomenCT-12organ
AbdomenCT12_TEMPLATE = {'liver': 1, 'rkidney': 2, 'spleen': 3, 'pancreas': 4, 'aorta': 5, 'postcava': 6,
                        'stomach': 7, 'gall': 8, 'esophagus': 9, 'radrenal': 10, 'ladrenal': 11, 'celiac_truck': 12,
                        'lkidney': 52}


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
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, data):
        y = data['label']

        # one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.long), num_classes=self.classes).permute(
            (-1, 1, 2, 3, 0)).squeeze(-1)

        # quit background class
        y = y[1:, :, :, :]

        data['label'] = y
        return data

