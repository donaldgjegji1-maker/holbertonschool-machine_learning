#!/usr/bin/env python3

SEED = 8

import os os.environ[PYTHONHASHSEED] = str(SEED) 
os.environ[TFENABLEONEDNNOPTS]= 0 
import random random.seed(SEED) 
import numpy as np np.random.seed(SEED) 
import tensorflow as tf tf.random.setseed(SEED)

buildmodel = _import(1-input).buildmodel optimizemodel = __import(2-optimize).optimizemodel 
onehot = import(3-onehot).onehot 
trainmodel = _import_(7-train).trainmodel

if name == main: 
    datasets = np.load(MNIST.npz) 
    Xtrain = datasets[Xtrain] 
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1) 
    Ytrain = datasets[Ytrain] 
    Ytrainoh = onehot(Ytrain) 
    Xvalid = datasets[Xvalid] 
    Xvalid = Xvalid.reshape(Xvalid.shape[0], -1) 
    Yvalid = datasets[Yvalid] 
    Yvalidoh = onehot(Yvalid)

lambtha = 0.0001
keep_prob = 0.95
network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
optimize_model(network, alpha, beta1, beta2)
batch_size = 64
epochs = 1000
train_model(network, X_train, Y_train_oh, batch_size, epochs,
            validation_data=(X_valid, Y_valid_oh), early_stopping=True,
            patience=3, learning_rate_decay=True, alpha=alpha)
