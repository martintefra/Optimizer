# Optimization for Machine Learning: Optimizer performance analysis
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



# Optimizer performance comparison project

## Overview
This project aims to analyze and compare the performance of various optimization algorithms on machine learning tasks. The analysis is conducted on two datasets: CIFAR-10 for image classification and the Adult dataset for tabular data classification.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)


**Install requirements**
To install the required packages, run the following command:

pip install -r requirements.txt



## Train and evaluate the models using the different optimizers
explain where and how to run ...
`python3 run.py --dataset <adult|cifar> `


**Note:**
You may need to adjust the variable `folder_path` to match your own environment.
