# autoencoder
Implementation of an autoencoder (https://en.wikipedia.org/wiki/Autoencoder). An autoencoder uses an encoder to compress the information.

# How to use it
## Train an autoencoder
It can use any keras model as encoder or decoder, they must comply with the following constraints:
1. The output of the decoder must have the same shape of the encoder input
2. The output of the encoder must have the same shape of the decoder input and that's the shape of the compressed representation

Here's a simple example using mnist image dataset:
```python
import autoencoder
from encoders import mnist_encoder
from decoders import mnist_decoder
import tensorflow as tf
import numpy as np

# Load and normalize mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
x_train_sample = x_train[0]

# Set the shape of the bottleneck
code_shape = (50,)

# Create encoder and decoder models
encoder = mnist_encoder(x_train_sample.shape, code_shape)
decoder = mnist_decoder(code_shape, x_train_sample.shape)

# Create autoencoder
mnist_autoencoder = autoencoder.Autoencoder(encoder, decoder)

# Training
mnist_autoencoder.train(x_train)

# Save trained models
mnist_autoencoder.save_models("test")
```
## Visualize results
I used this code to visualize the results:
```python
from matplotlib import pyplot as plt

# Pick a random image from the test set
original = x_test[np.random.randint(len(x_test))]

# Load encoder and decoder
encoder_name = 'encoder{}.hdf5'.format("test")
loaded_encoder = tf.keras.models.load_model(encoder_name)
decoder_name = 'decoder{}.hdf5'.format("test")
loaded_decoder = tf.keras.models.load_model(decoder_name)

# Encode the image
code = loaded_encoder.predict(original[None])

# Reconstruct the image
reconstructed = loaded_decoder.predict(code)

# Plot the original image, the reconstructed image and the pixel-wise difference between the two images
fig, axs = plt.subplots(1, 3)

axs[0].imshow(original, cmap=plt.cm.Greys)
axs[0].axis('off')
axs[0].title.set_text("Original")

axs[1].imshow(reconstructed[0], cmap=plt.cm.Greys)
axs[1].axis('off')
axs[1].title.set_text("Reconstructed")

axs[2].imshow(np.abs(original-reconstructed[0]), cmap=plt.cm.Greys)
axs[2].title.set_text("Difference")
axs[2].axis('off')

plt.show()
```
And this is what it looks like:

![Result Image](https://i.imgur.com/KRQSFMc.png)

## Experiment changing code size
I have changed the bottleneck size from 1 to 100 neurons and I'm reporting some visual results here. 
With only 1 neuron the decoder is not able to reconstruct the image:

![Code size 1](https://i.imgur.com/pMjvdzM.png)

Increasing the number of nodes to 3 the decoder is able to recreate a decent quality image but it does not look like the original number. The original number is a 3 while the reconstructed one looks more like a 5.

![Code size 3](https://i.imgur.com/XyO84bf.png)

When 10 neurons are used in the bottleneck layer the reconstructed image is of good quality and shows the correct number 3 but some details are still missing.

![Code size 10](https://i.imgur.com/vPNIzJr.png)

The quality of details increase by increasing the size of the bottleneck. Whit 100 neurons the reconstructed image contains most of the details of the original one even though there are still some differences between the two images.

![Code size 100](https://i.imgur.com/6WdEaF5.png)
