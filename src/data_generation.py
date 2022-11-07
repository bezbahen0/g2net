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
from scipy import stats

import pyfstat
from pyfstat.utils import get_sft_as_arrays

from .processing import processing_array


def fill_labels(output_path, output_csv_path, data_type, label_template, num_signals, target):
    id_ = [f"{data_type}/{label_template % i}" for i in range(num_signals)]

    label_pure = pd.DataFrame(data=id_, columns=["id"])
    label_pure["target"] = target

    Path(output_path).mkdir(parents=True, exist_ok=True)
    label_pure.to_csv(f"{output_csv_path}", index=False)


def save_hdf5(output_path, label, frequency, timestamps, fourier_data):
    f = h5py.File(f"{output_path}/{label}.hdf5", "w")
    g0 = f.create_group(f"PG{i}")
    g1 = g0.create_group("H1")
    g2 = g0.create_group("L1")
    dset = g0.create_dataset("frequency_Hz", data=frequency)
    g1.create_dataset("SFTs", data=fourier_data["H1"])
    g1.create_dataset("timestamps_GPS", data=timestamps["H1"])
    g2.create_dataset("SFTs", data=fourier_data["L1"])
    g2.create_dataset("timestamps_GPS", data=timestamps["L1"])
    f.close()


def save_data(writer, output_path, label, processing_function):
    frequency, timestamps, fourier_data = get_sft_as_arrays(writer.sftfilepath)

    if processing_function is None:
        save_hdf5(output_path, label, frequency, timestamps, fourier_data)
    else:
        processing_function(output_path, label, frequency, timestamps, fourier_data)

    for path in writer.sftfilepath.split(";"):
        os.remove(path)
    gc.collect()


# source - https://www.kaggle.com/code/horikitasaku/g2net-pyfstat-generate-pure-gaussian/notebook
def noise_generation(
    output_path, output_csv_path, data_type, num_signals, processing_function=None
):
    label_template = "noise_%i"

    fill_labels(output_path, output_csv_path, data_type, label_template, num_signals, 0)

    for i in tqdm(range(num_signals), leave=False, desc="Noise generation"):
        label = label_template % i
        # Setup Writer
        duration = 10357578 * np.random.uniform(1.5, 1.6)
        F0 = np.random.uniform(51, 497)
        sqrtSX = np.random.uniform(3e-24, 5e-24)
        tstart = 1238170021
        band = 1 / 5.01
        phi = np.random.uniform(1, -1) * np.pi

        noise_kwargs = {
            "label": label,
            "tstart": tstart,  # Starting time of the observation [GPS time]
            "duration": duration,  # Duration [seconds]
            "detectors": "H1,L1",  # Detector to simulate, in this case LIGO Hanford
            "F0": F0,  # Central frequency of the band to be generated [Hz]
            "Band": band,  # Frequency band-width around F0 [Hz]
            "sqrtSX": sqrtSX,  # Single-sided Amplitude Spectral Density of the noise
            "Tsft": 1800,  # Fourier transform time duration
            "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
            "SFTWindowBeta": 0.01,  # Parameter associated to the window function
            "phi": phi,
            "outdir": "/tmp/noise",
        }

        writer = pyfstat.Writer(**noise_kwargs)
        writer.make_data(verbose=False)

        save_data(writer, output_path, label, processing_function)


def signal_generation(
    output_path, output_csv_path, data_type, num_signals, processing_function=None
):
    label_template = "signal_%i"

    fill_labels(output_path, output_csv_path, data_type, label_template, num_signals, 1)

    writer_kwargs = {
        "label": f"signal",
        "tstart": 1238166018,
        "duration": 10357578 * np.random.uniform(1.5, 1.6),
        "detectors": "H1,L1",
        "sqrtSX": 1e-23,
        "Tsft": 1800,
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "Band": 0.2,
        "outdir": "/tmp/signal",
    }

    signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
        priors={
            "F0": {"uniform": {"low": 100.0, "high": 100.1}},
            "F1": -1e-10,
            "F2": 0,
            "h0": writer_kwargs["sqrtSX"] / 10,  # Fix amplitude at depth 10.
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
            "tref": writer_kwargs["tstart"],
        },
    )

    for i in tqdm(range(num_signals), leave=False, desc="Signal generation"):
        label = label_template % i
        # Draw signal parameters.
        # Noise can be drawn by setting `params["h0"] = 0
        params = signal_parameters_generator.draw()
        writer = pyfstat.Writer(**writer_kwargs, **params)
        writer.make_data(verbose=False)

        save_data(writer, output_path, label, processing_function)


def signal_with_noise_generation():
    pass


def main():
    """Runs data generation scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--num_signals", type=int)
    parser.add_argument("--processing", type=str, default=None)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--DATA GENERATION--")
    logger.info(f"config arguments: {args}")

    processing = None
    if args.processing == "baseline":
        processing = processing_array

    if args.data_type == "generated_noise":
        noise_generation(
            args.output, args.output_csv, args.data_type, args.num_signals, processing
        )

    if args.data_type == "generated_signal":
        signal_generation(
            args.output, args.output_csv, args.data_type, args.num_signals, processing
        )

    if args.data_type == "noise_signal":
        pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
