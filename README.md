# Kaggle Competition: G2Net Detecting Continuous Gravitational Waves

## Problem Statement
The goal of this competition is to develop a model capable of detecting weak and long-lived continuous gravitational wave signals emitted by rapidly rotating neutron stars in noisy data. The contest aims to help scientists detect a second class of gravitational waves, which could lead to further understanding of the structure of the most extreme stars in the Universe. The contest is designed to help detect signals from a class of gravitational waves that have not yet been detected and that could potentially provide new insights in the field.

## Data
You can get the download data from [here](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/data). Also if you want to run my code you need to put the raw unpacked data in ``data/raw`` directory

- ```[train/|test/]``` - folders containing the training and test files, files are presented in hdf5, and contain SFT (Short-time Fourier Transforms), spectrograms obtained from LIGO Livingston and LIGO Hanford
- ```train_labels.csv``` - a file containing the target labels. 1 if the data contains the presence of a gravitational wave, 0 otherwise. Target label - 1 was ignored, because the files with this label are just a [passcheck](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/363734) from the authors of the competition.
- ```sample_submission.csv``` - a sample submission file in the correct format.

## Approach
As a baseline project I used [Basic spectrogram image classification](https://www.kaggle.com/code/junkoda/basic-spectrogram-image-classification), and the main idea was to use the generation of new simulated data.

## Usage
Clone repo
```
git clone https://github.com/bezbahen0/g2net
```

Install requirements
```
pip install -r requirements.txt
```

And run generation new data, training, and inference.
```
snakemake --cores all
```

To run another experiment, you can replace the path of the configuration file in Snakefile, or change the configuration file located in the ```configs``` directory


## Results
| #  | Experiment | Coment      | Backend         | Input size | Private LB  | Public LB |
| -- | ---------- | --------------- | --------------- | ---------- | ------- | ---------| 
| 1  | baseline   | V3 data gen amplitued 20 | tf_efficientnet_b7_ns | 128*2  | 0.726  | 0.707    |
| 2  | spectorgram| augmentations, amplitued 20| tf_efficientnet_b7_ns | 128*2  | 0.745  | 0.721    |
| 3  | spectrogram| amplitued 30, augmentations | tf_efficientnet_b7_ns | 128*2  | 0.748  | 0.732    |
| 4  | spectrogram| linear layer, amplitude 30, dropout-0.25, lr-0.00056 | tf_efficientnet_b7_ns | 128*2  | 0.750  | 0.721    |