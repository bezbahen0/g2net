import os
import glob
import argparse
import logging

import torch

import pandas as pd
import numpy as np

from .dataset import Dataset
from .train import evaluate, get_model


def get_best_model(fold_data):
    checkpoints_names = list(fold_data['checkpoints'].keys())
    scores = [fold_data['checkpoints'][checkpoint]["score"] for checkpoint in checkpoints_names]
    index_max = max(range(len(scores)), key=scores.__getitem__)
    return fold_data['checkpoints'][checkpoints_names[index_max]]


def get_last_model(fold_data):
    checkpoints_names = list(fold_data['checkpoints'].keys())
    return fold_data['checkpoints'][checkpoints_names[-1]]


def predict(
    experiment,
    model_base_type,
    models_path,
    data_path,
    data_csv_path,
    submission_path,
    batch_size,
    n_workers,
    use_nfolds,
    use_best_model,
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
    for model_fold in checkpoint["folds"][:use_nfolds]:
        model = get_model(experiment, model_base_type, pretrained=False)

        checkpoint_epoch = (
            get_best_model(checkpoint[model_fold])
            if use_best_model
            else get_last_model(checkpoint[model_fold])
        )
        model.load_state_dict(checkpoint_epoch["model_params"])

        model.to(device)
        model.eval()

        test = evaluate(model, loader_test, device, compute_score=False, verbose=True)
        preds.append(test["y_pred"])

    submit["target"] = np.mean(preds, axis=0)

    submit.to_csv(submission_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base_type", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--models_path", type=str)
    parser.add_argument("--submission_path", type=str)
    parser.add_argument("--use_best_model", type=int)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_nfolds", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=6)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--INFERENCE MODEL--")
    logger.info(f"config arguments: {args}")

    predict(
        args.experiment,
        args.model_base_type,
        args.models_path,
        args.data_path,
        args.data_csv_path,
        args.submission_path,
        args.batch_size,
        args.n_workers,
        args.use_nfolds,
        args.use_best_model,
        args.device,
        logger,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
