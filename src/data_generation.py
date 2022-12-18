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

# from scipy import stats

import pyfstat
from pyfstat.utils import get_sft_as_arrays

from .config import Config
from .processing import get_processing_function, save_numpy

logging.getLogger('pyfstat').setLevel(logging.ERROR)

default_noise_kwargs = {
    "tstart": 1238170021,
    "duration": 4 * 30 * 86400,
    "Tsft": 1800,
    "Band": 0.2,
    "SFTWindowType": "tukey",
    "SFTWindowBeta": 0.001,
    "sqrtSX": 1e-23,
    "detectors": "H1,L1",
}


def fill_labels(
    output_path, output_csv_path, experiment, label_template, num_signals, target
):
    id_ = [f"{experiment}/{label_template % i}" for i in range(num_signals)]

    label_pure = pd.DataFrame(data=id_, columns=["id"])
    label_pure["target"] = target

    Path(output_path).mkdir(parents=True, exist_ok=True)
    label_pure.to_csv(f"{output_csv_path}", index=False)


def save_hdf5(output_path, label, frequency, timestamps, fourier_data):
    f = h5py.File(f"{output_path}/{label}.hdf5", "w")

    g0 = f.create_group(label)
    g1 = g0.create_group("H1")
    g2 = g0.create_group("L1")

    dset = g0.create_dataset("frequency_Hz", data=frequency)

    g1.create_dataset("SFTs", data=fourier_data["H1"])
    g1.create_dataset("timestamps_GPS", data=timestamps["H1"])

    g2.create_dataset("SFTs", data=fourier_data["L1"])
    g2.create_dataset("timestamps_GPS", data=timestamps["L1"])

    f.close()


def save_data(sftfilepath, output_path, label, processing):
    frequency, timestamps, fourier_data = get_sft_as_arrays(sftfilepath)
   
    if processing is None:
        save_hdf5(output_path, label, frequency, timestamps, fourier_data)
    else:
        processing_function, save_function = get_processing_function(processing)
        array = processing_function(frequency, timestamps, fourier_data)
        save_function(array, output_path, label)

    for path in sftfilepath.split(";"):
        os.remove(path)
    gc.collect()


def noise_sft_generation(label, tmp_dir):
    noise_kwargs = default_noise_kwargs.copy()
    noise_kwargs["label"] = label
    noise_kwargs["duration"] = noise_kwargs["duration"] * np.random.uniform(0.9, 1.0)
    noise_kwargs["sqrtSX"] = np.random.uniform(3e-24, 5e-24)
    noise_kwargs["F0"] = np.random.uniform(51, 497)
    noise_kwargs["outdir"] = tmp_dir

    writer = pyfstat.Writer(**noise_kwargs)
    writer.make_data(verbose=False)
    return writer, noise_kwargs


def noise_generation(
    output_path,
    output_csv_path,
    output_dir_name,
    config,
):
    label_template = "noise_%i"

    fill_labels(
        output_path, output_csv_path, output_dir_name, label_template, config.num_noise, 0
    )

    for i in tqdm(range(config.num_noise), leave=False, desc="Noise generation"):
        label = label_template % i
        writer, _ = noise_sft_generation(label, "/tmp/noise")
        save_data(writer.sftfilepath, output_path, label, config.processing)


def signal_generation(
    output_path,
    output_csv_path,
    output_dir_name,
    config
):
    label_template = "signal_%i"

    fill_labels(
        output_path, output_csv_path, output_dir_name, label_template, config.num_signals, 1
    )

    for i in tqdm(range(config.num_signals), leave=False, desc="Signal generation"):
        label = label_template % i

        noise_writer, noise_kwargs = noise_sft_generation(label, "/tmp/signal")

        priors = {
            "F0": noise_kwargs["F0"],
            "F1": -1e-10,
            "F2": 0,
            "h0": noise_kwargs["sqrtSX"] / config.sqrtsxdiv,  # 20.
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
            "tref": noise_kwargs["tstart"],
            "SFTWindowType": "tukey",
            "SFTWindowBeta": 0.001,
            "noiseSFTs": noise_writer.sftfilepath,
        }

        signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
            priors=priors
        )

        # Draw signal parameters.
        # Noise can be drawn by setting `params["h0"] = 0
        params = signal_parameters_generator.draw()

        writer = pyfstat.Writer(**params)
        writer.make_data(verbose=False)

        for path in noise_writer.sftfilepath.split(";"):
            os.remove(path)
        gc.collect()

        save_data(writer.sftfilepath, output_path, label, config.processing)


def main():
    """Runs data generation scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--config_path", type=str)
    
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--DATA GENERATION--")

    config = Config()
    config.load_config(args.config_path, logger)
    
    if args.data_type == "generated_noise":
        noise_generation(
            args.output,
            args.output_csv,
            f"{args.data_type}_{config.processing}_{config.num_noise}_{config.sqrtsxdiv}",
            config
        )

    if args.data_type == "generated_signal":
        signal_generation(
            args.output,
            args.output_csv,
            f"{args.data_type}_{config.processing}_{config.num_signals}_{config.sqrtsxdiv}",
            config
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
