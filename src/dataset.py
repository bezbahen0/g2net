import os

import cv2
import torch
import torchaudio

import numpy as np
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        df,
        config,
        augmentation=False,
    ):
        self.data_path = data_path
        self.df = df
        self.augmentation = config.augmentaion_train and augmentation
        self.flip_rate = config.flip_rate
        self.freq_shift_rate = config.freq_shift_rate
        self.time_mask_num = config.time_mask_num
        self.freq_mask_num = config.freq_mask_num
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
        filename = f"{self.data_path}/{file_id}.npz"
        with np.load(filename) as data:
            img = data["arr_0"]
        img = img.squeeze()

        if self.augmentation:
            img = np.flip(img, axis=1).copy()
            img = np.flip(img, axis=2).copy()
            img = np.roll(
                img, np.random.randint(low=0, high=img.shape[1]), axis=1
            ).copy()

            img = torch.from_numpy(img)
            for _ in range(np.random.randint(low=0, high=self.time_mask_num)):
                img = self.transforms_time_mask(img)

            for _ in range(np.random.randint(low=0, high=self.freq_mask_num)):
                img = self.transforms_freq_mask(img)

        return img, y

class ImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        df,
        config,
        augmentation=False,
    ):
        super().__init__(
            data_path,
            df,
            config,
            augmentation=False,
        )
        self.img_size = config.input_size

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id
        filename = f"{self.data_path}/{file_id}"
        channels = []
        for channel, channel_name in enumerate(["H1", "L1"]):
            channels.append(cv2.imread(f"{filename}/{channel_name}.png", cv2.IMREAD_GRAYSCALE)[:,:self.img_size])

        img = np.stack(np.asarray(channels)) / 255.0
        
        if self.augmentation:
            img = np.flip(img, axis=1).copy()
            img = np.flip(img, axis=2).copy()
            img = np.roll(
                img, np.random.randint(low=0, high=img.shape[1]), axis=1
            ).copy()

            img = torch.from_numpy(img)
            for _ in range(np.random.randint(low=0, high=self.time_mask_num)):
                img = self.transforms_time_mask(img)

            for _ in range(np.random.randint(low=0, high=self.freq_mask_num)):
                img = self.transforms_freq_mask(img)
        img = torch.from_numpy(img).float()
        return img, y

class DatasetAllSignal(Dataset):
    def __init__(
        self,
        data_path,
        df,
        config,
        augmentation=False,
    ):
        super().__init__(
            data_path,
            df,
            config,
            augmentation=False,
        )
        self.num_splits = config.num_splits

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id
        filename = f"{self.data_path}/{file_id}.npz"
        with np.load(filename) as data:
            img = data["arr_0"]

        img = img.squeeze()
        #img = img[:, :, :128]

        if self.augmentation:
            img = np.flip(img, axis=1).copy()
            img = np.flip(img, axis=2).copy()
            img = np.roll(
                img, np.random.randint(low=0, high=img.shape[1]), axis=1
            ).copy()

            img = torch.from_numpy(img)
            for _ in range(np.random.randint(low=0, high=self.time_mask_num)):
                img = self.transforms_time_mask(img)

            for _ in range(np.random.randint(low=0, high=self.freq_mask_num)):
                img = self.transforms_freq_mask(img)
        img = torch.from_numpy(img)
        splited = torch.split(img, 128, 2)
        result = splited[:self.num_splits]
        result = torch.cat(result)
        return img, y


def get_dataset_class(dataset_name):
    if dataset_name == "baseline":
        return Dataset
    elif dataset_name == "spectrogram":
        return ImageDataset
    elif dataset_name == "all-signal":
        return DatasetAllSignal
    else:
        raise NotImplementedError("Dataset is not implemented")
