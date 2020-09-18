# Microsoft Stock Price Prediction using LSTM (Long-Short-Term-Memory)

## Table of Contetnts
* [General-Info](#general-info)
* [Technologies](#technologies)

## General Info

The project will showcase the prediction of Microsoft Stock Price. For this project we will implement RNN (Recurent-Neural-Network). This project will predict prices before COVID-19 crisis, because we will predict the trend of the stock price. And we know that the model can't adjust to spikes. 

## Technologies
* Python 3.7.7 (https://www.python.org/)
* Jupyter Notebook (https://jupyter.org/)
* Visual Studio Code (https://code.visualstudio.com/)
* TensorFlow 2.1.0 (https://www.tensorflow.org/)
* Git & GitHub (https://github.com/)
* Windows Subsystem for Linux (WSL2) (https://docs.microsoft.com/en-us/windows/wsl/wsl2-index)

This week (15.09.2020) the Jupyter Notebook for VS Code broke down, so we will continue in 7 days when the next "Recovery" build will be released
Update: They fixed it on 16.09.2020

## The Project

This will not be a typical Deep Learning Project where we have thons of data and then we need to preprocess it, prepare it for the DL algorithm and then train and test the model.

Our data is already in good shape, we do not have any missing data and the only phases of preprocessing for this project are:
1. What Variables do we want to use for the model?
2. Scale the data
3. Create the 3D Structure for the algorithm


We got our data from Yahoo Finance (finance.yahoo.com). We devided our data into:
1. Training set (source/MSFT_TRAIN.csv)
2. Testing set (source/MSFT_TEST.csv)

Our training set has 2676 rows and 7 columns and contains data from: 04.01.2010 - 19.08.2020. For our training set we have the remaining values which range from 20.08.2020. - 18.09.2020. and we have here 20 financial days.

Next question to ask is how much days we want to train on to predict the next day? This is a very good question and we would answer 50 - 60 in normal times, but in the time of the COVID-19 pandemic we would say 2 - 5 days. And this answer comes as I was training my predicted trade courve was always so terible. I was thinking how much do I overfit my model, implementing the EarlyStopping mechanism 2 days of experimenting passed and I finnaly found my answer and I'm happy to present it to you.

Next up is to select our Variables that we will use to predict the prices. I will use the Open Stock Price to predict Open Stock Price

We answered the first question, the next question is how do we scale the data. For that is recommended to use the <b>MinMaxScaler</b> from sklearn.preprocessing.

And the last question is the 3D data structure to use. I will link every single thing that I'm talking about here in the Refferences section.
To create the data structure we need numpy. Is really simple:

1. What array we want to reshape
2. The lenght of the array (array.shape[0])
3. The number of columnse (array.shape[1])
4. The number of predictors 

<code>xTrain = np.reshape(1.(2. , 3.), 4.)</code> 

For us as you will see in the notebook is like this:

<code>xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))</code>

We do the same for the Test dataSet because we will use the test dataset as our validation when we setup EarlyStopping.

Now to our model, the model will consist of:
* One input LSTM layer,
* Two hidden LSTM layers,
* One output dense layer

For the loss will use MSE(mean_squared_error) and for our optimizer will use adam. 

Now we need to fit our model and setup EarlyStopping and you will find that in the notebook.

# References

EarlyStopping(https://keras.io/api/callbacks/early_stopping/)

LSTM(https://keras.io/api/layers/recurrent_layers/lstm/)

MinMaxScaler(https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)