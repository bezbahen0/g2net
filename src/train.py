import os
import time
import torch
import logging
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .dataset import Dataset
from .models.model import Model


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
    model_name,
    model_save_path,
    nfold,
    epochs,
    batch_size,
    n_workers,
    weight_decay,
    max_grad_norm,
    lr_max,
    epochs_warmup,
    random_state,
    device,
    logger,
):
    kfold = StratifiedKFold(n_splits=nfold, random_state=random_state, shuffle=True)

    df = pd.read_csv(data_csv_path)
    dataset = Dataset(data_path, df)

    models = {"kfold": nfold, "folds": [], "epochs": epochs, "scores": []}
    for ifold, (idx_train, idx_test) in enumerate(kfold.split(dataset, df["target"])):
        logger.info("Fold %d/%d" % (ifold, nfold))
        torch.manual_seed(42 + ifold + 1)

        # Train - val split
        dataset_train = Dataset(data_path, df.iloc[idx_train])
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
        model = Model(model_name, pretrained=True)
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(
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

        tb = time.time()
        logger.info("Epoch   loss          score   lr")
        for iepoch in range(epochs):
            loss_sum = 0.0
            n_sum = 0

            train_losses = []
            val_losses = []
            scores = []
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

            train_losses.append(loss_train)
            val_losses.append(val["loss"])
            scores.append(val["score"])

            logger.info(
                "Epoch %d %.4f %.4f %.4f  %.2e  %.2f sec"
                % (iepoch + 1, loss_train, val["loss"], val["score"], lr_now, dt)
            )

        dt = time.time() - tb
        logger.info("Training done %.2f sec total, %.2f sec val" % (dt, time_val))

        # Save fold model params
        fold_name = f"model_fold_{ifold}"
        val = evaluate(model, loader_val, device)

        models[f"model_fold_{ifold}"] = {
            "model": model.state_dict(),
            "score": val["score"],
            "loss": val["loss"],
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "scores": scores,
        }

        models["folds"].append(fold_name)
        models["scores"].append(val["score"])
        del model, optimizer
        torch.cuda.empty_cache()

    logger.info(f"Average CV score: {np.mean(models['scores'])}")
    models["cv_score"] = np.mean(models["scores"])

    torch.save(models, model_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_csv_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--model_type", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=int, default=1000)
    parser.add_argument("--lr_max", type=float, default=4e-4)
    parser.add_argument("--epochs_warmup", type=float, default=1.0)
    parser.add_argument("--random_state", type=float, default=42)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")
    logger.info(f"config arguments: {args}")

    train(
        args.data_path,
        args.data_csv_path,
        args.model_type,
        args.model_save_path,
        args.nfold,
        args.epochs,
        args.batch_size,
        args.n_workers,
        args.weight_decay,
        args.max_grad_norm,
        args.lr_max,
        args.epochs_warmup,
        args.random_state,
        args.device,
        logger,
    )

    logger.info(f"Save trained {args.model_type} model to {args.model_save_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
