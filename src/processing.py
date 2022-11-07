import os
import h5py
import logging
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from multiprocessing import Pool

def processing_array(output_path, filename, frequency, timestamps, fourier_data):
    img = np.empty((2, 360, 128), dtype=np.float32)

    for ch, s in enumerate(["H1", "L1"]):
        a = fourier_data[s][:360, :4096] * 1e22  # Fourier coefficient complex64
        p = a.real**2 + a.imag**2  # power
        p /= np.mean(p)  # normalize
        p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128
        img[ch] = p

        np.save(os.path.join(output_path, f"{filename}.npy"), img)

def processing_chunk(args):
    chunk_data, chunk_id, data_path, output_path, mode = args

    pbar = tqdm(
        chunk_data.iterrows(),
        desc=f"Processing {chunk_id} chunk in {mode} mode",
        total=len(chunk_data),
        position=chunk_id,
        leave=False,
    )

    for id, (file_id, label) in pbar:
        y = np.float32(label)
        img = np.empty((2, 360, 128), dtype=np.float32)

        filename = f"{data_path}/{file_id}.hdf5"
        with h5py.File(filename, "r") as f:
            g = f[file_id]
            for ch, s in enumerate(["H1", "L1"]):
                a = g[s]["SFTs"][:360, :4096] * 1e22  # Fourier coefficient complex64
                p = a.real**2 + a.imag**2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128
                img[ch] = p

        np.save(os.path.join(output_path, f"{file_id}.npy"), img)


def processing_pool(
    data_path, data_csv_path, mode, output_path, output_csv, n_workers, logger
):
    df = pd.read_csv(data_csv_path)
    df = df[df.target >= 0]

    Path(output_path).mkdir(parents=True, exist_ok=True)

    args = []
    for chunk_id, chunk_data in enumerate(np.array_split(df, n_workers)):
        function_args = [chunk_data, chunk_id, data_path, output_path, mode]
        args.append(function_args)

    pool = Pool(processes=n_workers)
    block = pool.map(processing_chunk, args)

    df['id'] = df['id'].apply(lambda id: f'{mode}/{id}')
    df.to_csv(output_csv)


def main():
    """Runs data processing scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--data_csv", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--n_workers", type=int)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--PROCESSING--")
    logger.info(f"config arguments: {args}")

    processing_pool(
        args.data,
        args.data_csv,
        args.mode,
        args.output,
        args.output_csv,
        args.n_workers,
        logger,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
