# Multitask Neural Network for Broad Spectral Feature Detection

Author: Priyam Das  
Email: priyam.das@unsw.edu.au  

## Overview

This repository contains a dense neural network framework designed to assist astronomers in detecting and fitting broad spectral features in IFU datasets (optimized for MUSE NFM AO mode).

The architecture performs:

- Classification (0, 1, 2 peaks)
- Regression (peak centroid prediction)
- Guided Gaussian fitting
- Doppler shift extraction
- Surface brightness estimation
- Velocity width calculation

## Architecture

Input → 128 (ReLU) → Dropout → 64 (ReLU) → Dropout  
Two output heads:
- Softmax (classification)
- Sigmoid (peak positions)

## Features

- Synthetic spectrum generation
- Multitask neural network training
- Automated Gaussian fitting with fallback logic
- Parallel datacube processing

## Requirements

See `requirements.txt`

## Usage

Modify the parameters in the main section of `fit_curve`.
Total sample of synthetic spectra is 100000 and batch_size is 20. Change these according to your requirement.
Customize the name of the output files in run_ml.py. 
Use .json file to define simulation and detection parameters. Copy or edit the values in one of the json files according to your requirement.
The code is not tested on narrow emission lines or any absorption lines. However, I believe the code can classify and detect two narrow lines if the sigma values (width of the lines) are adjusted accordingly

use below in bash:
python run_ml.py --fits_file your_file.fits --params_file params_sulphur.json --x_start 0 --x_end 200 --y_start 0 --y_end 200

