import os

import torch
import torchaudio

import numpy as np
import pandas as pd



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        df,
        augmentation=False,
        flip_rate=0.0,
        freq_shift_rate=0.0,
        time_mask_num=0,
        freq_mask_num=0,
    ):
        self.data_path = data_path
        self.df = df
        self.augmentation = augmentation
        self.flip_rate = flip_rate
        self.freq_shift_rate = freq_shift_rate
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.transforms_time_mask = torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )

        self.transforms_freq_mask = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id
        filename = f"{self.data_path}/{file_id}.npy"
        img = np.load(filename)

        if self.augmentation:
            img = np.flip(img, axis=1).copy()
            img = np.flip(img, axis=2).copy()
            img = np.roll(img, np.random.randint(low=0, high=img.shape[1]), axis=1).copy()

            for _ in range(np.random.randint(low=0, high=self.time_mask_num)): 
                img = self.transforms_time_mask(img)
            for _ in range(np.random.randint(low=0, high=self.freq_mask_num)):
                img = self.transforms_freq_mask(img)

        return img, y
