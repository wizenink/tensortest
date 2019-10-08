from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import configparser
import time
import matplotlib.pyplot as plt
import numpy as np

import models
import train
import settings
from rgb2yuv import *


import cProfile

settings.init()
print("RUNNING WITH TENSORFLOW VERSION ", tf.__version__)




_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
# _URL = r'http://chaladze.com/l5/img/Linnaeus%205%20256X256.rar'



path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
print(path_to_zip)
PATH = os.path.join(os.path.dirname(path_to_zip), 'lin/')


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image,tf.float32)
    image = tf.divide(image,255.0)
    image = tf.image.rgb_to_yuv(image)  
    image = tf.reshape(image,(IMG_WIDTH,IMG_HEIGHT,3))

    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    #channels = tf.unstack(image,axis=-1)
    #image = tf.stack([channels[2],channels[1],channels[0]],axis=-1)
    input_image = tf.expand_dims(image[:,:,0],-1)
    real_image = image[:,:,1:3]

    #input_image = tf.cast(input_image,tf.float32)
    #input_image = tf.divide(input_image,255.0)
    #real_image = tf.cast(real_image,tf.float32)
    #real_image = tf.divide(real_image,255.0)

    
    return input_image,real_image



#train_dataset = tf.data.Dataset.list_files(PATH+'train/*/*.jpg')
train_dataset = tf.data.Dataset.list_files(settings.config['paths']['train_dataset'])
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(settings.config.getint('training','batch_size'))

test_dataset = tf.data.Dataset.list_files(settings.config['paths']['test_dataset'])
#test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load,num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(settings.config.getint('training','batch_size'))


logdir = os.path.join(settings.config.get('paths','tb_logs'),settings.config.get('paths','log_tag'))
writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()
pr = cProfile.Profile()

train.fit(train_dataset,test_dataset,settings.config.getint('training','epochs'))
