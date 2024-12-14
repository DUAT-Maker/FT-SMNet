## FT-SMNet-pytorch
This is a basic description of the pytorch version of FT-SMNet. This project was submitted to Mechanical Systems and Signal Processing under the title: "FT-SMNet: Fourier transform sparse matrix network for structural health monitoring time series data forecasting"

# FT-SMNet
-------------

> FT-SMNet is a temporal transformation network that utilizes Fourier Transform and sparse matrices. The network primarily consists of two fundamental modules: FT and Sparse Matrix block.

## Data:  
Datasets are located in the datasets folder.

## Usage
Batch prediction for data FuXing.csv:
'''
sh scripts/exp_ori.sh
'''
Real-time prediction for data FuXing.csv:
'''
sh scripts/exp_real_time.sh
'''
Batch prediction for data FuXing_MAV.csv:
'''
sh scripts/exp_MAV.sh
'''
## Model Performance

![Prediction Results](figs/results.png)
For more information, please refer to the article.
