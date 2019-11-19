import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def mnist_decoder(code_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(256, input_shape=code_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(output_shape), activation='sigmoid'))
    model.add(layers.Reshape(output_shape))
    
    code_layer = layers.Input(shape=code_shape)
    output = model(code_layer)
    
    return models.Model(code_layer, output)
