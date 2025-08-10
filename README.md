# Handwriting Digit Recognition

This project implements a machine learning pipeline to identify handwritten digits using the MNIST dataset. The code in this folder is designed to train and test models that classify images of handwritten digits (0-9).

## Features

- Uses the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv) for both training and testing.
- Preprocessing and feature extraction for image data.
- Model training and evaluation for digit classification.
- Jupyter Notebooks and Python scripts for experimentation and reproducibility.

## Folder Structure

- `train.ipynb` &mdash; Main notebook for training and evaluating the model.
- `predict.ipynb` &mdash; Main notebook for training and evaluating the model.
- `data/` &mdash; Place to store the MNIST dataset files.

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- jupyter

## Usage

Before starting ensure to unzip the `mnist_train.csv.zip`. This file will act as the training data source for the neural network.

To train the neural network, open the `train.ipynb` file and run all the cells. This will generate a `trained_params.npz` file in the `bin` folder. This file hold the trained parameters i.e. the weight and bias matrix which can then be used in `predict.ipynb` to test the results.

## License

MIT License
