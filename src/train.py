import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .models import get_model_class
from .dataset import get_dataset_class


def evaluate(model, loader_val, device, *, compute_score=True, verbose=False):
    """
    Predict and compute loss and score
    """
    tb = time.time()
    was_training = model.training
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []

    if verbose:
        pbar = tqdm(desc="Predict", total=len(loader_val))

    for img, y in loader_val:

        n = y.size(0)
        img = img.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(img)

        loss = criterion(y_pred.view(-1), y)

        n_sum += n
        loss_sum += n * loss.item()

        y_all.append(y.cpu().detach().numpy())
        y_pred_all.append(y_pred.sigmoid().squeeze().cpu().detach().numpy())

        if verbose:
            pbar.update()

    loss_val = loss_sum / n_sum

    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)

    score = roc_auc_score(y, y_pred) if compute_score else None

    ret = {
        "loss": loss_val,
        "score": score,
        "y": y,
        "y_pred": y_pred,
        "time": time.time() - tb,
    }

    model.train(was_training)  # back to train from eval if necessary

    return ret


def train(
    data_path,
    data_csv_path,
    config,
    logger,
):
    kfold = StratifiedKFold(
        n_splits=config.nfold, random_state=config.random_state, shuffle=True
    )

    df = pd.read_csv(data_csv_path)
    dataset_class = get_dataset_class(config.dataset)
    dataset = dataset_class(data_path, df, config, augmentation=False)

    models = {"folds": []}
    for ifold, (idx_train, idx_test) in enumerate(kfold.split(dataset, df["target"])):
        logger.info(f"Fold {ifold}/{config.nfold}")
        torch.manual_seed(config.random_state + ifold + 1)

        # Train - val split
        dataset_train = dataset_class(
            data_path, df.iloc[idx_train], config, config.augmentaion_train
        )

        dataset_val = dataset_class(data_path, df.iloc[idx_test], config)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            num_workers=config.n_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=config.batch_size,
            num_workers=config.n_workers,
            pin_memory=True,
        )

        # Model and optimizer
        model = get_model_class(config.model_name)
        model = model(config)
        model.to(config.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        # Learning-rate schedule
        nbatch = len(loader_train)
        warmup = config.epochs_warmup * nbatch  # number of warmup steps
        nsteps = config.epochs * nbatch  # number of total steps

        scheduler = CosineLRScheduler(
            optimizer,
            warmup_t=warmup,
            warmup_lr_init=0.0,
            warmup_prefix=True,  # 1 epoch of warmup
            t_initial=(nsteps - warmup),
            lr_min=1e-6,
        )  # 3 epochs of cosine

        #scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #    optimizer=optimizer,
        #    max_lr=config.lr_max,
        #    total_steps=len(dataset_train) * config.epochs,
        #    pct_start=config.once_cycle_pct_start,
        #)

        time_val = 0.0
        lrs = []

        model_checkpoints = {}

        tb = time.time()
        logger.info("Epoch   loss          score   lr")
        for iepoch in range(config.epochs):
            loss_sum = 0.0
            n_sum = 0

            # Train
            pbar = tqdm(
                loader_train, desc=f"Train epoch {iepoch}", total=len(loader_train)
            )
            for ibatch, (img, y) in enumerate(pbar):
                n = y.size(0)
                img = img.to(config.device)
                y = y.to(config.device)

                optimizer.zero_grad()

                y_pred = model(img)
                loss = criterion(y_pred.view(-1), y)

                loss_train = loss.item()
                loss_sum += n * loss_train
                n_sum += n

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()

                scheduler.step(iepoch * nbatch + ibatch + 1)
                lrs.append(optimizer.param_groups[0]["lr"])

            # Evaluate
            val = evaluate(model, loader_val, config.device)

            time_val += val["time"]

            loss_train = loss_sum / n_sum
            lr_now = optimizer.param_groups[0]["lr"]

            dt = time.time() - tb

            model_checkpoints[f"checkpoint_{iepoch}"] = {
                "loss": loss_train,
                "val_loss": val["loss"],
                "score": val["score"],
                "model_params": model.state_dict(),
                "oof_predictions": val["y_pred"],
                "oof_targets": val["y"],
                "epoch": iepoch,
                "epoch_lr": lr_now,
            }

            torch.save(model_checkpoints, f"/tmp/model_fold_{ifold}_checkpoints.pt")

            logger.info(
                f"Epoch {iepoch} {loss_train} {val['loss']} {val['score']}  {lr_now}  {dt} sec"
            )

        dt = time.time() - tb
        logger.info(f"Training done {dt} sec total, {time_val} sec val")

        # Save fold model params
        fold_name = f"model_fold_{ifold}"

        models[f"model_fold_{ifold}"] = {"checkpoints": model_checkpoints}

        models["folds"].append(fold_name)
    return models


def add_training_meta(model, config):
    model["config"] = config.__dict__
    model["write_time"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")

    config = Config()
    config.load_config(args.config_path, logger)

    model = train(
        args.data_path,
        args.data_csv_path,
        config,
        logger=logger,
    )
    model = add_training_meta(model, args)
    torch.save(model, args.model_save_path)

    logger.info(f"Save trained {config.model_name} model to {args.model_save_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
