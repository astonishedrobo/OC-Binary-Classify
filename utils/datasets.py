'''
Normalization part in the dataloader adapted from https://github.com/EIDOSLAB/torchstain
'''


import glob
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import pandas as pd
from torchvision import transforms
import torchstain

class BioClassify(Dataset):
    def __init__(self, paths = {"data": None}):
        self.data = pd.read_csv(paths["data"])
        self.target_stain = paths["target_stain"]

    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.data.loc[idx, "path"]), cv2.COLOR_BGR2RGB)
        label = self.data.loc[idx, "label"]
        # target_stain = cv2.cvtColor(cv2.imread(self.target_stain), cv2.COLOR_BGR2RGB)

        # T = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x*255)
        # ])

        # normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        # normalizer.fit(T(target_stain))

        # t_to_transform = T(img)
        # norm = normalizer.normalize(I=t_to_transform).numpy()
        norm = img

        norm = cv2.resize(norm, (299, 299))
        norm = torch.as_tensor(norm)
        norm = F.convert_image_dtype(norm, torch.float)
        label = torch.as_tensor(label)

        return norm, label