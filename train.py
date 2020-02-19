import tensorflow as tf
from models import *
from metrics import *
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import settings
EPOCHS = 100



def generate_images(model, test_input, tar,epoch):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    noise = tf.random.uniform([test_input.shape[0],256,256,1])
    prediction = model([noise,test_input], training=True)
    plt.imsave(f"results/image-at-epoch-{epoch}-u.png",prediction[0][:,:,0])
    plt.imsave(f"results/image-at-epoch-{epoch}-v.png",prediction[0][:,:,1])
    print(test_input[0].shape)
    
    final = np.concatenate((test_input[0],prediction[0]),axis=2)
    final = tf.image.yuv_to_rgb(final)
    final = np.array(final,dtype=np.float)
    final = np.clip(final,0.0,1.0)
    print(final.shape)
    plt.imsave(f"results/image-at-epoch-{epoch}-rgb.jpg",final)


def generate_plots(g_loss_mean,d_loss_mean,epochs):
    plt.figure()
    plt.plot(epochs,g_loss_mean,'g',label='Generator loss')
    plt.plot(epochs,d_loss_mean,'b',label='Discriminator loss')
    plt.savefig(settings.config['paths']['plots']+'losses.png')

@tf.function
def train_step(noise,input_image,target,generator_optimizer):

    with tf.GradientTape() as gen_tape:
        
        gen_output = generator([noise,input_image],training=True)

        g_loss = gen_loss(gen_output,target)

        generator_gradients = gen_tape.gradient(g_loss,generator.trainable_variables) 
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
        tf.summary.histogram("generator gradients",generator_gradients[0],step=generator_optimizer.iterations)
        return g_loss

def fit(ds,tds,epochs):
    start_epoch = 0
    
    generator_optimizer = tf.keras.optimizers.Adam(settings.config.getfloat('training','lr_generator'),settings.config.getfloat('training','beta1_generator'))
    g_loss_mean = []
    epochlist = []
    for epoch in range(start_epoch,epochs):
        start = time.time()
        g_losses = []
        avg_g_loss = tf.keras.metrics.Mean(name='g_loss',dtype=tf.float32)
        i = 0
        for input_image,target in ds:
            noise = tf.random.uniform([settings.config.getint('training','batch_size'),256,256,1])
            
            g_loss = train_step(noise,input_image,target,generator_optimizer)
            avg_g_loss.update_state(g_loss)
            if tf.equal(generator_optimizer.iterations % 10,0):
                tf.summary.scalar('g_loss',avg_g_loss.result(),step=generator_optimizer.iterations)
                avg_g_loss.reset_states()
            g_losses.append(g_loss)
            print("batch={}".format(i),end='\r')
            i = i+1
        print(f"Finished epoch {epoch}")
        #tf.summary.trace_export(name='train_func',step=0,profiler_outdir=os.path.join(settings.config.get('paths','tb_logs'),settings.config.get('paths','log_tag')))
        for example_input, example_target in tds.take(1):
            image = generate_images(generator, example_input, example_target,epoch)
            #tf.summary.image("result_epoch{:04d}".format(epoch),tf.expand_dims(image,0),step=epoch)
        print(f"Finished saving image in epoch {epoch}")

        #tf.summary.scalar("epoch_time",time.time()-start,step=epoch)
        #epochlist.append(epoch)
        #g_loss_mean.append(np.array(g_losses).mean())
        #print('Epoch[{}/{}] - {}s - G:{}'.format(epoch+1,epochs,time.time()-start,g_loss_mean[epoch]))

