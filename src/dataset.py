import os
import torch
import h5py
import numpy as np
import pandas as pd
from icecream import ic

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

        img = np.empty((2, 360, 128), dtype=np.float32)

        filename = f'{self.data_path}/{self.data_type}/{file_id}.hdf5'
        with h5py.File(filename, 'r') as f:
            g = f[file_id]

            for ch, s in enumerate(['H1', 'L1']):
                a = g[s]['SFTs'][:, :4096] * 1e22  # Fourier coefficient complex64

                p = a.real**2 + a.imag**2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128

                img[ch] = p

        return img, y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join("../data/raw/g2net-detecting-continuous-gravitational-waves/", "train_labels.csv"))
    dataset = Dataset('train', "../data/raw/g2net-detecting-continuous-gravitational-waves/", df)
    
    img, y = dataset[10]

    plt.figure(figsize=(8, 3))
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.imshow(img[0, 300:360])  # zooming in for dataset[10]
    plt.colorbar()
    plt.show()