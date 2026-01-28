#!/usr/bin/env python3

SEED = 8

import os os.environ[PYTHONHASHSEED] = str(SEED) 
os.environ[TFENABLEONEDNNOPTS]= 0 
import random random.seed(SEED) 
import numpy as np np.random.seed(SEED) 
import tensorflow as tf tf.random.setseed(SEED)


onehot = _import(3-onehot).onehot 
trainmodel = _import(8-train).trainmodel 
model = _import__(9-model)

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

network = model.load_model('network1.keras')
batch_size = 32
epochs = 1000
train_model(network, X_train, Y_train_oh, batch_size, epochs,
            validation_data=(X_valid, Y_valid_oh), early_stopping=True,
            patience=2, learning_rate_decay=True, alpha=0.001)
model.save_model(network, 'network2.keras')
network.summary()
print(network.get_weights())
del network

network2 = model.load_model('network2.keras')
network2.summary()
print(network2.get_weights())
