import os
import h5py
import logging
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

def processing(data_path, data_csv_path, output_path, output_csv, mode, logger):
    df = pd.read_csv(data_csv_path)
    df = df[df.target >= 0]

    Path(os.path.join(output_path,f'{mode}')).mkdir(parents=True, exist_ok=True)
    for id, (file_id, label) in tqdm(df.iterrows(), desc="Processing", total=len(df)):
        y = np.float32(label)

        img = np.empty((2, 360, 128), dtype=np.float32)

        filename = f'{data_path}/{mode}/{file_id}.hdf5'
        with h5py.File(filename, 'r') as f:
            g = f[file_id]

            for ch, s in enumerate(['H1', 'L1']):
                a = g[s]['SFTs'][:, :4096] * 1e22  # Fourier coefficient complex64

                p = a.real**2 + a.imag**2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128
                
                img[ch] = p

        np.save(os.path.join(output_path,f'{mode}/{file_id}.npy'), img)
        
    df.to_csv(output_csv)


def main():
    """Runs data processing scripts
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--data_csv", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--mode", type=str)
    
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--PROCESSING--")
    logger.info(f"config arguments: {args}")

    processing(args.data, args.data_csv, args.output, args.output_csv, args.mode, logger)
    

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()