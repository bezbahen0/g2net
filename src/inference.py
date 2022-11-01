import os
import glob
import argparse
import logging

import torch

import pandas as pd
import numpy as np

from .models.model import Model
from .dataset import Dataset
from .train import evaluate


def predict(
    model_name,
    models_path,
    data_path,
    data_csv_path,
    submission_path,
    batch_size,
    n_workers,
    device,
    logger,
):
    submit = pd.read_csv(data_csv_path)
    dataset_test = Dataset(data_path, submit)
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, num_workers=n_workers, pin_memory=True
    )

    checkpoint = torch.load(models_path)
    preds = []
    for model_fold in checkpoint["folds"]:
        model = Model(model_name, pretrained=False)
        model.to(device)
        model.load_state_dict(checkpoint[model_fold]['model'])
        model.eval()

        test = evaluate(model, loader_test, device, compute_score=False, verbose=True)
        preds.append(test["y_pred"])

    submit["target"] = np.mean(preds, axis=0)
    
    submit.to_csv(submission_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--models_path", type=str)
    parser.add_argument("--submission_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=6)
    # parser.add_argument("--quantile", type=float)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--INFERENCE MODEL--")
    logger.info(f"config arguments: {args}")

    predict(
        args.model_name,
        args.models_path,
        args.data_path,
        args.data_csv_path,
        # args.quantile,
        args.submission_path,
        args.batch_size,
        args.n_workers,
        args.device,
        logger,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
