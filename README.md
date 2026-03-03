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
run python run_ml.py --fits_file --params_file params_sulphur.json --x_start 0 --x_end 200 --y_start 0 --y_end 200
