import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def mnist_encoder(input_shape, code_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(np.prod(code_shape), activation='sigmoid'))
    model.add(layers.Reshape(code_shape))
    
    input_layer = layers.Input(shape=input_shape)
    code = model(input_layer)
    
    return models.Model(input_layer, code)

