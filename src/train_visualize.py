import logging
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize(model_path, logger):
    model = torch.load(model_path)

    figure, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    for i, fold in enumerate(model['folds']):
        fold_data = model[fold]
        epochs = model['epochs']

        ax1.plot(range(epochs), fold_data['train_loss_history'], label=f'fold-{i}')
        
        ax2.plot(range(epochs), fold_data['val_loss_history'], label=f'fold-{i}')
        
        ax3.plot(range(epochs), fold_data['scores'], label=f'fold-{i}')
        
  
    ax1.set_title("Train loss per epoch")
    #ax1.ylabel("train loss")

    ax2.set_title("Validation loss per epoch")
    #ax2.ylabel("val loss")

    ax3.set_title("Scores per epoch")
    #ax3.ylabel("Score")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--VISUALIZE TRAIN--")
    logger.info(f"config arguments: {args}")

    visualize(args.model_path, logger)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()