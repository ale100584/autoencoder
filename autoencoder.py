import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class Autoencoder():
    '''
    Class to implement and train an autoencoder [https://en.wikipedia.org/wiki/Autoencoder]

    Input:
    - encoder: keras model that will be used for encoding
    - decoder: keras model that will be used for decoding
    - code_shape: shape of the bottleneck of the autoencoder
    '''

    def __init__(self, encoder, decoder, code_shape=None):
        # If code_shape is not specified then code_shape is set as encoder output shape
        if code_shape is None:
            code_shape = encoder.output.shape[1:]
        # Input checks
        if not(encoder.input.shape[1:] == decoder.output.shape[1:]):
            raise Exception("encoder input shape doesn't match decoder output shape")
        if not(encoder.output.shape[1:] == code_shape):
            raise Exception("encoder output shape doesn't match bottleneck shape (code_shape)")
        if not(decoder.input.shape[1:] == code_shape):
            raise Exception("decoder input shape doesn't match bottleneck shape (code_shape)")

        self.encoder = encoder
        self.decoder = decoder
        self.code_shape = code_shape
        
        input_layer = layers.Input(shape=encoder.input.shape[1:])
        code = self.encoder(input_layer)
        output = self.decoder(code)
    
        self.autoencoder = models.Model(input_layer, output)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, X_train, X_test=None, epochs=10):
        '''
        Trains the autoencoder using X_train as training data. If provided, it uses X_test as validation data.
    
        Input:
        - X_train: training set
        - X_test: test set
        - epochs: number of epochs
        '''

        validation_data = None
        if X_test is not None:
            validation_data = [X_test, X_test]
        self.autoencoder.fit(x=X_train, y=X_train, epochs=epochs, validation_data = validation_data)

    def save_models(self, suffix=""):
        '''
        Saves encoder and autoencoder to file. The output files will be encoder<suffix>.hdf5 and decoder<suffix>.hdf5

        Input:
        - suffix: suffix appended to encoder and decoder filenames
        '''

        self.encoder.save('encoder{}.hdf5'.format(suffix))
        self.decoder.save('decoder{}.hdf5'.format(suffix))
