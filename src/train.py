import os
import time
import logging
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .dataset import Dataset
from .models.baseline import BaselineModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_model(experiment, model_base_name, pretrained=False):
    if experiment == "baseline":
        return BaselineModel(model_base_name, pretrained=pretrained)
    else:
        raise NotImplementedError("Model not implemented")


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
    experiment,
    model_name,
    nfold,
    epochs,
    batch_size,
    n_workers,
    weight_decay,
    max_grad_norm,
    lr_max,
    epochs_warmup,
    random_state,
    augmentaion_train,
    flip_rate,
    freq_shift_rate,
    time_mask_num,
    freq_mask_num,
    pretrained,
    device,
    logger,
):
    kfold = StratifiedKFold(n_splits=nfold, random_state=random_state, shuffle=True)

    df = pd.read_csv(data_csv_path)
    dataset = Dataset(data_path, df)

    models = {"folds": []}
    for ifold, (idx_train, idx_test) in enumerate(kfold.split(dataset, df["target"])):
        logger.info(f"Fold {ifold}/{nfold}")
        torch.manual_seed(random_state + ifold + 1)

        # Train - val split
        dataset_train = Dataset(
            data_path,
            df.iloc[idx_train],
            augmentaion_train,
            flip_rate,
            freq_shift_rate,
            time_mask_num,
            freq_mask_num,
        )
        dataset_val = Dataset(data_path, df.iloc[idx_test])

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, num_workers=n_workers, pin_memory=True
        )

        # Model and optimizer
        model = get_model(experiment, model_name, pretrained=pretrained)
        model.to(device)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr_max, weight_decay=weight_decay
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        # Learning-rate schedule
        nbatch = len(loader_train)
        warmup = epochs_warmup * nbatch  # number of warmup steps
        nsteps = epochs * nbatch  # number of total steps

        scheduler = CosineLRScheduler(
            optimizer,
            warmup_t=warmup,
            warmup_lr_init=0.0,
            warmup_prefix=True,  # 1 epoch of warmup
            t_initial=(nsteps - warmup),
            lr_min=1e-6,
        )  # 3 epochs of cosine

        time_val = 0.0
        lrs = []

        model_checkpoints = {}

        tb = time.time()
        logger.info("Epoch   loss          score   lr")
        for iepoch in range(epochs):
            loss_sum = 0.0
            n_sum = 0

            # Train
            for ibatch, (img, y) in enumerate(loader_train):
                n = y.size(0)
                img = img.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_pred = model(img)
                loss = criterion(y_pred.view(-1), y)

                loss_train = loss.item()
                loss_sum += n * loss_train
                n_sum += n

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()

                scheduler.step(iepoch * nbatch + ibatch + 1)
                lrs.append(optimizer.param_groups[0]["lr"])

            # Evaluate
            val = evaluate(model, loader_val, device)

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
            }

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


def add_training_meta(model, args):
    config = vars(args)
    model["config"] = config
    model["write_time"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--model_base_type", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--pretrained", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=int, default=1000)
    parser.add_argument("--lr_max", type=float, default=4e-4)
    parser.add_argument("--epochs_warmup", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--augmentaion_train", type=int, default=1)
    parser.add_argument("--flip_rate", type=float, default=0.5)
    parser.add_argument("--freq_shift_rate", type=float, default=1.0)
    parser.add_argument("--time_mask_num", type=int, default=1)
    parser.add_argument("--freq_mask_num", type=int, default=2)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")
    logger.info(f"config arguments: {args}")

    set_seed(args.random_state)

    model = train(
        args.data_path,
        args.data_csv_path,
        args.experiment,
        args.model_base_type,
        args.nfold,
        args.epochs,
        args.batch_size,
        args.n_workers,
        args.weight_decay,
        args.max_grad_norm,
        args.lr_max,
        args.epochs_warmup,
        args.random_state,
        args.augmentaion_train,
        args.flip_rate,
        args.freq_shift_rate,
        args.time_mask_num,
        args.freq_mask_num,
        pretrained=args.pretrained,
        device=args.device,
        logger=logger,
    )
    model = add_training_meta(model, args)
    torch.save(model, args.model_save_path)

    logger.info(f"Save trained {args.experiment} model to {args.model_save_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
