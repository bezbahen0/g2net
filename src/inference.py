import os
import glob
import argparse
import logging

import torch

import pandas as pd
import numpy as np

from .train import evaluate
from .config import Config
from .models import get_model_class
from .dataset import get_dataset_class


def get_best_model(fold_data):
    checkpoints_names = list(fold_data["checkpoints"].keys())
    scores = [
        fold_data["checkpoints"][checkpoint]["score"]
        for checkpoint in checkpoints_names
    ]
    index_max = max(range(len(scores)), key=scores.__getitem__)
    return fold_data["checkpoints"][checkpoints_names[index_max]]


def get_last_model(fold_data):
    checkpoints_names = list(fold_data["checkpoints"].keys())
    return fold_data["checkpoints"][checkpoints_names[-1]]


def predict(
    models_path,
    data_path,
    data_csv_path,
    submission_path,
    config,
    logger,
):
    submit = pd.read_csv(data_csv_path)
    dataset_class = get_dataset_class(config.dataset)
    dataset_test = dataset_class(data_path, submit, config)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.test_batch_size,
        num_workers=config.n_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(models_path)
    preds = []
    for model_fold in checkpoint["folds"][: config.use_nfolds]:
        model = get_model_class(config.model_name)
        model = model(config)

        checkpoint_epoch = (
            get_best_model(checkpoint[model_fold])
            if config.use_best_models
            else get_last_model(checkpoint[model_fold])
        )
        model.load_state_dict(checkpoint_epoch["model_params"])

        model.to(config.device)
        model.eval()

        test = evaluate(
            model, loader_test, config.device, compute_score=False, verbose=True
        )
        preds.append(test["y_pred"])

    submit["target"] = np.mean(preds, axis=0)

    submit.to_csv(submission_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--models_path", type=str)
    parser.add_argument("--submission_path", type=str)
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--INFERENCE MODEL--")

    config = Config()
    config.load_config(args.config_path, logger)

    predict(
        args.models_path,
        args.data_path,
        args.data_csv_path,
        args.submission_path,
        config,
        logger,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
