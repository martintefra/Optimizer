# Optimizer Performance Study on Neural Networks



## Overview
This project aims to analyze and compare the performance of various optimization algorithms on machine learning tasks. The analysis is conducted using two distinct datasets: CIFAR-10 for image classification and the Adult dataset for tabular data classification. By evaluating the impact of different optimizers, including Lion, SGD, SignSGD, Adagrad, and Adam, on both simple and complex models, we provide insights into their strengths and weaknesses. Our study includes modifications such as layer-wise learning rates to assess performance and convergence speed, ultimately guiding the selection of the most effective optimizer for different data types and model complexities.

## Code Structure
```
Optimizer/
|
├── datasets/
│   ├── cifar/
│   └── adult.csv
|
├── models/
│   ├── Adult_NN.py
│   └── Cifar_NN.py
|
├── optimizers/
│   ├── Lion.py
│   └── SignSGD.py
|
├── plots/
│   ├── adult/
|   |   └── loss_plots.png
│   └── cifar/
|       └── loss_plots.png
|
├── README.md
├── requirements.txt
├── run.py
└── utils.py
```

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)


**Install requirements**
To install the required packages, run the following command:

```sh 
pip install -r requirements.txt
```



## Train and evaluate the models using the different optimizers
Run the code on a specific dataset and optionally choose the complex model (default: simple).

```sh
python3 run.py --dataset <adult|cifar> [--complex]
```


**Note:**
You may need to adjust the variable `folder_path` to match your own environment.

## About
Optimizer was developed as the project of the MSc. level course of EPFL: "Optimization for machine learning"
