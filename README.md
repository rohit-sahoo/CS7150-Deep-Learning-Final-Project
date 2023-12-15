# CS7150-Deep-Learning-Final-Project
CS 7150: Deep Learning Fall 2023 Final Project

## An Analysis of "Long-term Forecasting with TiDE: Time-series Dense Encoder"
### Analysis by: 
Rohit Sisir Sahoo (NUID: 002736531, Email: sahoo.ro@northeastern.edu)
Pratik Satish Hotchandani (NUID:  , Email: hotchandani.p@northeastern.edu)

### Abstract: 
Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.

### Code Format:

1. File: time_series_prediction.py

Content:
Defines and trains two time series forecasting models (NHiTS and TiDE) on the Australian Beer Consumption dataset.
Utilizes PyTorch Lightning for training, early stopping, and model evaluation.
Plots the input, ground truth, and predictions of both models for visual comparison.
Compares the performance of the models using Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics in a bar chart.
Purpose:

Demonstrates time series forecasting using NHiTS and TiDE models, evaluating and comparing their performance on the Australian Beer Consumption dataset.

![Results 1](https://drive.google.com/file/d/1dRPJCESfUe_o_WJfmZv25KLhTo9A0QSx)

![Results 2](https://drive.google.com/file/d/1dRPJCESfUe_o_WJfmZv25KLhTo9A0QSx/view?usp=drive_link)

2. File: train.py

Content:
Main training code for time series forecasting using TensorFlow and Keras.
Supports various configuration options through command-line flags.
Utilizes a custom TideModel with deep neural networks for training.
Implements early stopping based on validation loss to prevent overfitting.
Saves best model predictions and metrics for later evaluation.
Purpose:

Demonstrates training a time series forecasting model with configurable options and early stopping based on validation loss.

3. File: models.py

Content:
Defines a neural network model (MLPResidual, _make_dnn_residual, TideModel) for time-series prediction.
Contains various utility functions for evaluation metrics (mape, mae_loss, wape, smape, rmse, nrmse) and a dictionary of metrics (METRICS).

Purpose:
Implementing a multi-scale deep neural network model for time-series forecasting.

4. File: data_loader.py

Content:
Defines a data loader class (TimeSeriesdata) for loading time-series data from a CSV file.
Contains methods for generating training, validation, and test data batches.
Purpose:
Loading and preparing time-series data for training and evaluation.

5. File: time_covariates.py

Content:
Defines a class TimeCovariates for extracting time-related covariates.
Calculates minute of hour, hour of day, day of week, day of month, day of year, month of year, week of year covariates.
Optionally fetches holiday features and calculates the distance to various holidays.
Purpose:
Provides functionality to extract time-related covariates for time-series data.

