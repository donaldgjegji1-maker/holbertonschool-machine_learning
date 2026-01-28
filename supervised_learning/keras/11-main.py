#!/usr/bin/env python3

SEED = 8

import os os.environ[PYTHONHASHSEED] = str(SEED) 
os.environ[TFENABLEONEDNNOPTS]= 0 
import random random.seed(SEED) 
import numpy as np np.random.seed(SEED) 
import tensorflow as tf tf.random.setseed(SEED)
model = import(9-model) 
config = import(11-config)

if name == main: 
    network = model.loadmodel(network1.keras) 
    config.saveconfig(network, config1.json) 
    del network

network2 = config.load_config('config1.json')
network2.summary()
print(network2.get_weights())
