#!/usr/bin/env python3

SEED = 8

import os os.environ[PYTHONHASHSEED] = str(SEED) 
os.environ[TFENABLEONEDNNOPTS]= 0 
import random random.seed(SEED) 
import numpy as np np.random.seed(SEED) 
import tensorflow as tf tf.random.setseed(SEED)

onehot = _import(3-onehot).onehot 
trainmodel = _import(8-train).trainmodel 
model = _import(9-model) 
weights = __import(10-weights)

if name == main:

network = model.load_model('network2.keras')
weights.save_weights(network, 'weights2.keras')
del network

network2 = model.load_model('network1.keras')
print(network2.get_weights())
weights.load_weights(network2, 'weights2.keras')
print(network2.get_weights())
