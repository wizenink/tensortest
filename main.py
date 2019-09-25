from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()
import os
import time
import matplotlib.pyplot as plt


import models
import train



_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
# _URL = r'http://chaladze.com/l5/img/Linnaeus%205%20256X256.rar'



path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
print(path_to_zip)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:,:w,:]
    input_image = image[:,w:,:]

    input_image = tf.cast(input_image,tf.float32)
    input_image = tf.divide(input_image,255.0)
    real_image = tf.cast(real_image,tf.float32)
    real_image = tf.divide(real_image,255.0)

    return input_image,real_image


train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(8)



train.fit(train_dataset,100)