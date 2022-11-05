import gc
import os
import sys
import logging
import argparse

import h5py

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

import pyfstat
from pyfstat.utils import get_sft_as_arrays

#logger = pyfstat.set_up_logger(log_level="INFO", outdir="/tmp/gen_log", append=False, streams=(open(os.devnull, 'w'),))


# source - https://www.kaggle.com/code/horikitasaku/g2net-pyfstat-generate-pure-gaussian/notebook
def noise_generation(output_path, output_csv_path, num_signals):
    id_ = [
        f"PG{i}" for i in range(num_signals)
    ]  # Create a label.csv   PG is id's name. the num in range is the id

    label_pure = pd.DataFrame(data=id_, columns=["id"])
    label_pure["target"] = 0

    Path(output_path).mkdir(parents=True, exist_ok=True)
    label_pure.to_csv(f"{output_csv_path}", index=False)


    for i in tqdm(range(num_signals), leave=False, desc="Data generation"):
        # Setup Writer
        duration = 10357578 * np.random.uniform(1.5, 1.6)
        F0 = np.random.uniform(51, 497)
        sqrtSX = np.random.uniform(3e-24, 5e-24)
        tstart = 1238170021
        band = 0.09972222222221
        phi = np.random.uniform(1, -1) * np.pi

        noise_kwargs = {
            "label": "gaussian_noise_h1_l1_detectors",
            "tstart": tstart,  # Starting time of the observation [GPS time]
            "duration": duration,  # Duration [seconds]
            "detectors": "H1,L1",  # Detector to simulate, in this case LIGO Hanford
            "F0": F0,  # Central frequency of the band to be generated [Hz]
            "Band": band,  # Frequency band-width around F0 [Hz]
            "sqrtSX": sqrtSX,  # Single-sided Amplitude Spectral Density of the noise
            "Tsft": 3600,  # Fourier transform time duration
            "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
            "SFTWindowBeta": 0.01,  # Parameter associated to the window function
            "phi": phi,
            "outdir": "/tmp/noise",
        }

        noise_writer = pyfstat.Writer(**noise_kwargs)
        noise_writer.make_data(verbose=False)

        frequency, timestamps, fourier_data = get_sft_as_arrays(
            noise_writer.sftfilepath
        )

        f = h5py.File(f"{output_path}/PG{i}.hdf5", "w")
        g0 = f.create_group(f"PG{i}")
        g1 = g0.create_group("H1")
        g2 = g0.create_group("L1")
        dset = g0.create_dataset("frequency_Hz", data=frequency)

        g1.create_dataset("SFTs", data=fourier_data["H1"])
        g1.create_dataset("timestamps_GPS", data=timestamps["H1"])

        g2.create_dataset("SFTs", data=fourier_data["L1"])
        g2.create_dataset("timestamps_GPS", data=timestamps["L1"])

        f.close()

        for path in noise_writer.sftfilepath.split(";"):
            os.remove(path)
        gc.collect()


def signal_generation():
    pass


def signal_with_noise_generation():
    pass


def main():
    """Runs data generation scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--num_signals", type=int)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--DATA GENERATION--")
    logger.info(f"config arguments: {args}")


    if args.data_type == "noise":
        noise_generation(args.output, args.output_csv, args.num_signals)

    if args.data_type == "signal":
        pass

    if args.data_type == "noise_signal":
        pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
