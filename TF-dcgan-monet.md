
##### Copyright 2019 The TensorFlow Authors.


```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Deep Convolutional Generative Adversarial Network

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/generative/dcgan.ipynb">
    <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
    View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

This tutorial demonstrates how to generate images of handwritten digits using a [Deep Convolutional Generative Adversarial Network](https://arxiv.org/pdf/1511.06434.pdf) (DCGAN). The code is written using the [Keras Sequential API](https://www.tensorflow.org/guide/keras) with a `tf.GradientTape` training loop.

## What are GANs?
[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GANs) are one of the most interesting ideas in computer science today. Two models are trained simultaneously by an adversarial process. A *generator* ("the artist") learns to create images that look real, while a *discriminator* ("the art critic") learns to tell real images apart from fakes.

![A diagram of a generator and discriminator](https://tensorflow.org/alpha/tutorials/generative/images/gan1.png)

During training, the *generator* progressively becomes better at creating images that look real, while the *discriminator* becomes better at telling them apart. The process reaches equilibrium when the *discriminator* can no longer distinguish real images from fakes.

![A second diagram of a generator and discriminator](https://tensorflow.org/alpha/tutorials/generative/images/gan2.png)

This notebook demonstrate this process on the MNIST dataset. The following animation shows a series of images produced by the *generator* as it was trained for 50 epochs. The images begin as random noise, and increasingly resemble hand written digits over time.

![sample output](https://tensorflow.org/images/gan/dcgan.gif)

To learn more about GANs, we recommend MIT's [Intro to Deep Learning](http://introtodeeplearning.com/) course.





### Import TensorFlow and other libraries


```python
from __future__ import absolute_import, division, print_function, unicode_literals
```


```python
!pip install tensorflow-gpu==2.0.0-alpha0

```

    Requirement already satisfied: tensorflow-gpu==2.0.0-alpha0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (2.0.0a0)
    Requirement already satisfied: keras-applications>=1.0.6 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.0.7)
    Requirement already satisfied: numpy<2.0,>=1.14.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.15.4)
    Requirement already satisfied: protobuf>=3.6.1 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (3.7.1)
    Requirement already satisfied: astor>=0.6.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (0.7.1)
    Requirement already satisfied: tb-nightly<1.14.0a20190302,>=1.14.0a20190301 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.14.0a20190301)
    Requirement already satisfied: absl-py>=0.7.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (0.7.1)
    Requirement already satisfied: grpcio>=1.8.6 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.20.1)
    Requirement already satisfied: tf-estimator-nightly<1.14.0.dev2019030116,>=1.14.0.dev2019030115 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.14.0.dev2019030115)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.0.9)
    Requirement already satisfied: termcolor>=1.1.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.1.0)
    Requirement already satisfied: wheel>=0.26 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (0.31.1)
    Requirement already satisfied: six>=1.10.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (1.11.0)
    Requirement already satisfied: gast>=0.2.0 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (0.2.2)
    Requirement already satisfied: google-pasta>=0.1.2 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tensorflow-gpu==2.0.0-alpha0) (0.1.6)
    Requirement already satisfied: h5py in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from keras-applications>=1.0.6->tensorflow-gpu==2.0.0-alpha0) (2.8.0)
    Requirement already satisfied: setuptools in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from protobuf>=3.6.1->tensorflow-gpu==2.0.0-alpha0) (39.1.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tb-nightly<1.14.0a20190302,>=1.14.0a20190301->tensorflow-gpu==2.0.0-alpha0) (0.14.1)
    Requirement already satisfied: markdown>=2.6.8 in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from tb-nightly<1.14.0a20190302,>=1.14.0a20190301->tensorflow-gpu==2.0.0-alpha0) (3.1)
    [31mfastai 1.0.50.post1 requires nvidia-ml-py3, which is not installed.[0m
    [31mthinc 6.12.1 has requirement msgpack<0.6.0,>=0.5.6, but you'll have msgpack 0.6.0 which is incompatible.[0m
    [33mYou are using pip version 10.0.1, however version 19.1.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
import tensorflow as tf
```


```python
tf.__version__
```




    '2.0.0-alpha0'




```python
# To generate GIFs
!pip install imageio
```

    Requirement already satisfied: imageio in /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (2.3.0)
    [31mfastai 1.0.50.post1 requires nvidia-ml-py3, which is not installed.[0m
    [31mthinc 6.12.1 has requirement msgpack<0.6.0,>=0.5.6, but you'll have msgpack 0.6.0 which is incompatible.[0m
    [33mYou are using pip version 10.0.1, however version 19.1.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
```

### Load and prepare the dataset



```python
# ADDED TO ACCESS S3 DATA

import boto3
import sys
from boto.s3.key import Key

client = boto3.client('s3') #low-level functional API

resource = boto3.resource('s3') #high-level object-oriented API
my_bucket = resource.Bucket('monet-monet') #subsitute this for your s3 bucket name. 

# ADDED TO INTERATE THROUGH S3 DATA
# iteratate through s3 bucket

for key in my_bucket.objects.all():
    print(key.key)
```


```python
(train_images, train_labels), (_, _) = my_bucket
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-75-eb117d76d66c> in <module>()
    ----> 1 (train_images, train_labels), (_, _) = my_bucket
    

    TypeError: 's3.Bucket' object is not iterable



```python
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
```


```python
BUFFER_SIZE = 60000
BATCH_SIZE = 256
```


```python
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

## Create the models

Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

### The Generator

The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.


```python
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
```

Use the (as yet untrained) generator to create an image.


```python
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
```

### The Discriminator

The discriminator is a CNN-based image classifier.


```python
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
```

Use the (as yet untrained) discriminator to classify the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.


```python
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
```

## Define the loss and optimizers

Define loss functions and optimizers for both models.



```python
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

### Discriminator loss

This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.


```python
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
```

### Generator loss
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.


```python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

The discriminator and the generator optimizers are different since we will train two networks separately.


```python
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

### Save checkpoints
This notebook also demonstrates how to save and restore models, which can be helpful in case a long running training task is interrupted.


```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```

## Define the training loop




```python
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
```

The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.


```python
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```


```python
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
```

**Generate and save images**




```python
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
```

## Train the model
Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.


```python
%%time
train(train_dataset, EPOCHS)
```

Restore the latest checkpoint.


```python
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

## Create a GIF



```python
# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
```


```python
display_image(EPOCHS)
```

Use `imageio` to create an animated gif using the images saved during training.


```python
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)
```

If you're working in Colab you can download the animation with the code below:


```python
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download(lite_model_path)
```

## Next steps


This tutorial has shown the complete code necessary to write and train a GAN. As a next step, you might like to experiment with a different dataset, for example the Large-scale Celeb Faces Attributes (CelebA) dataset [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset/home). To learn more about GANs we recommend the [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).

