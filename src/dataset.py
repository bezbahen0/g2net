import os
import torch

import numpy as np
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(data_type, df)

    img, y = dataset[i]
      img (np.float32): 2 x 360 x 128
      y (np.float32): label 0 or 1
    """
    def __init__(self, data_type, data_path, df):
        self.data_type = data_type
        self.data_path = data_path
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id
        filename = f'{self.data_path}/{self.data_type}/{file_id}.npy'
        img = np.load(filename)       

        return img, y