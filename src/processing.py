import os
import h5py
import logging
import argparse

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from scipy.stats import norm
from multiprocessing import Pool

from .config import Config


def normalize(X):
    X = (X[..., None].view(X.real.dtype) ** 2).sum(-1)
    POS = int(X.size * 0.99903)
    EXP = norm.ppf((POS + 0.4) / (X.size + 0.215))
    scale = np.partition(X.flatten(), POS, -1)[POS]
    X /= scale / EXP.astype(scale.dtype) ** 2
    return X


def rescale(input, H1, L1):
    input = torch.from_numpy(input)
    rescale = torch.tensor([[H1, L1]])
    tta = torch.randn([1, *input.shape, 2], dtype=torch.float32).square_().sum(-1)
    tta *= rescale[..., None, None] / 2
    valid = ~torch.isnan(input)
    tta[:, valid] = input[valid].float()
    return tta


def processing_large_kernel(frequency, timestamps, fourier_data):
    astime = np.full([2, 360, 5760], np.nan, dtype=np.float32)

    HT = (np.asarray(timestamps["H1"]) / 1800).round().astype(np.int64)
    LT = (np.asarray(timestamps["L1"]) / 1800).round().astype(np.int64)

    MIN = min(HT.min(), LT.min())
    HT -= MIN
    LT -= MIN

    H1 = normalize(np.asarray(fourier_data["H1"], np.complex128))
    valid = HT < 5760
    astime[0][:, HT[valid]] = H1[:360, valid]

    L1 = normalize(np.asarray(fourier_data["L1"], np.complex128))
    valid = LT < 5760
    astime[1][:, LT[valid]] = L1[:360, valid]

    return rescale(astime, H1.mean(), L1.mean())


def save_numpy(img, output_path, filename):
    # with gzip.GzipFile(os.path.join(output_path, f"{filename}.npz.gz"), "w") as f:
    #    np.save(file=f, arr=img)
    np.savez_compressed(os.path.join(output_path, f"{filename}.npz"), img)


def processing_baseline(frequency, timestamps, fourier_data):
    img = np.empty((2, 360, 128), dtype=np.float32)

    for ch, s in enumerate(["H1", "L1"]):
        a = fourier_data[s][:360, :4096] * 1e22  # Fourier coefficient complex64
        p = a.real**2 + a.imag**2  # power
        p /= np.mean(p)  # normalize
        p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128
        img[ch] = p

    return img


def processing_temp(frequency, timestamps, fourier_data):
    return np.empty((2, 360, 128), dtype=np.float32)


def get_processing_function(processing_name):
    if processing_name == "baseline":
        return processing_baseline
    elif processing_name == "large-kernel":
        return processing_large_kernel
    else:
        raise NotImplementedError("Preprocessing data function not implemented")


def processing_chunk(args):
    chunk_data, chunk_id, data_path, output_path, mode, config = args

    pbar = tqdm(
        chunk_data.iterrows(),
        desc=f"Processing {chunk_id} chunk in {mode} mode",
        total=len(chunk_data),
        position=chunk_id,
        leave=False,
    )

    processing_function = get_processing_function(config.processing)

    for id, (file_id, label) in pbar:
        y = np.float32(label)

        filename = f"{data_path}/{file_id}.hdf5"
        with h5py.File(filename, "r") as f:
            g = f[file_id]
            fourier_data = {"H1": g["H1"]["SFTs"], "L1": g["L1"]["SFTs"]}
            frequency = g["frequency_Hz"]
            timestamps = {
                "H1": g["H1"]["timestamps_GPS"],
                "L1": g["L1"]["timestamps_GPS"],
            }

            img = processing_function(frequency, timestamps, fourier_data)
            save_numpy(img, output_path, file_id)


def processing_pool(
    data_path,
    data_csv_path,
    mode,
    output_path,
    output_csv,
    config,
    logger,
):
    df = pd.read_csv(data_csv_path)
    df = df[df.target >= 0]

    Path(output_path).mkdir(parents=True, exist_ok=True)

    args = []
    for chunk_id, chunk_data in enumerate(np.array_split(df, 1)):  # config.n_workers
        function_args = [
            chunk_data,
            chunk_id,
            data_path,
            output_path,
            mode,
            config,
        ]
        args.append(function_args)

    pool = Pool(processes=1)  # config.n_workers
    block = pool.map(processing_chunk, args)

    df["id"] = df["id"].apply(lambda id: f"{mode}/{id}")
    df.to_csv(output_csv)


def main():
    """Runs data processing scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--data_csv", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--PROCESSING--")

    config = Config()
    config.load_config(args.config_path, logger)

    processing_pool(
        args.data,
        args.data_csv,
        args.mode,
        args.output,
        args.output_csv,
        config,
        logger,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
