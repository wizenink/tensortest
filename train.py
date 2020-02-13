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
    prediction = model(test_input, training=True)
    
    plt.figure(figsize=(15,15))


    print("Test input shape:",test_input.shape)
    print("Target shape:",tar.shape)
    real_output = np.concatenate((test_input[0],tar[0]),axis=2)
    #real_output = real_output[...,::-1]

    gen_output = np.concatenate((test_input[0],prediction[0]),axis=2)
    #gen_output = gen_output[...,::-1]


    #display_list = [test_input[0], tar[0], prediction[0]]
    
    display_list = [
        real_output[:,:,0],
        gen_output[:,:,0],
        real_output[:,:,1],
        gen_output[:,:,1],
        real_output[:,:,2],
        gen_output[:,:,2],
        tf.image.yuv_to_rgb(real_output),
        tf.image.yuv_to_rgb(gen_output) 
    ]
    #display_list = [real_output,gen_output]

    print("GENERATED VALUES: Min:{} Max:{}".format(np.amin(display_list[1]),np.amax(display_list[1])))
    title = [
        'Ground Truth Y',
        'Generated Y',
        'Ground Truth U',
        'Generated U',
        'Ground Truth V',
        'Generated V',
        'Ground Truth RGB',
        'Generated RGB'
        ]
    x = 0
    for i in range(2):
        for j in range(4):
            print("adding image to subplot: ",((j+i*4)+1))
            plt.subplot(4, 2,(j+i*4)+1)
            plt.title(title[x])
            # getting the pixel values between [0, 1] to plot it.
            if(len(display_list[x].shape) == 2):
                plt.imshow(display_list[x],cmap='gray')
            else:
                plt.imshow(display_list[x])
            plt.axis('off')
            x = x+1
    plt.savefig(settings.config['paths']['results']+'image_at_epoch_{:04d}.png'.format(epoch))

    return plt.imread(settings.config['paths']['results']+'image_at_epoch_{:04d}.png'.format(epoch))
    #cv2.imwrite('result_{:04d}.png'.format(epoch),cv2.cvtColor(gen_output,cv2.COLOR_YUV2BGR))
    
    #plt.show()

def generate_plots(g_loss_mean,d_loss_mean,epochs):
    plt.figure()
    plt.plot(epochs,g_loss_mean,'g',label='Generator loss')
    plt.plot(epochs,d_loss_mean,'b',label='Discriminator loss')
    plt.savefig(settings.config['paths']['plots']+'losses.png')
@tf.function
def train_step(noise,input_image,target,generator_optimizer,discriminator_optimizer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_output = generator(input_image,training=False)

        disc_real = discriminator([input_image,target],training=True)
        disc_gen = discriminator([input_image,gen_output],training=True)

        #g_loss = gen_loss(disc_gen,gen_output,target)
        d_loss_fake,d_loss_real = disc_loss(disc_real,disc_gen)
        discriminator_gradients = disc_tape.gradient(d_loss_fake+d_loss_real,discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

        disc_gen = discriminator([input_image,gen_output],training=False)
        gen_output = generator(input_image,training=True)
        g_loss = gen_loss(disc_gen,gen_output,target)
        #d_loss_fake,d_loss_real = disc_loss(disc_real,disc_gen)

        generator_gradients = gen_tape.gradient(g_loss,generator.trainable_variables) 
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
        tf.summary.histogram("generator gradients",generator_gradients[0],step=discriminator_optimizer.iterations)
        tf.summary.histogram("discriminator_gradients",discriminator_gradients[0],step=discriminator_optimizer.iterations)
        return g_loss,d_loss_fake,d_loss_real

def fit(ds,tds,epochs):
    start_epoch = 0
    
    generator_optimizer = tf.keras.optimizers.Adam(settings.config.getfloat('training','lr_generator'),settings.config.getfloat('training','beta1_generator'))
    discriminator_optimizer = tf.keras.optimizers.Adam(settings.config.getfloat('training','lr_discriminator'),settings.config.getfloat('training','beta1_discriminator'))

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
    

    manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = int(settings.config['training']['start_epoch'])
    else:
        print("Initializing from scratch.")
    g_loss_mean = []
    d_loss_mean = []
    epochlist = []
    for epoch in range(start_epoch,epochs):
        start = time.time()
        g_losses = []
        d_losses = []
        avg_g_loss = tf.keras.metrics.Mean(name='g_loss',dtype=tf.float32)
        avg_d_loss_real = tf.keras.metrics.Mean(name='d_loss_real',dtype=tf.float32)
        avg_d_loss_fake = tf.keras.metrics.Mean(name='d_loss_fake',dtype=tf.float32)
        #tf.summary.trace_on(graph=True,profiler=True)
        for input_image,target in ds:
            noise = tf.random.uniform([settings.config.getint('training','batch_size'),256,256,1])
            
            g_loss,d_loss_fake,d_loss_real = train_step(noise,input_image,target,generator_optimizer,discriminator_optimizer)
            avg_g_loss.update_state(g_loss)
            avg_d_loss_fake.update_state(d_loss_fake)
            avg_d_loss_real.update_state(d_loss_real)
            if tf.equal(generator_optimizer.iterations % 10,0):
                tf.summary.scalar('g_loss',avg_g_loss.result(),step=generator_optimizer.iterations)
                tf.summary.scalar('d_loss_fake',avg_d_loss_fake.result(),step=discriminator_optimizer.iterations)
                tf.summary.scalar('d_loss_real',avg_d_loss_real.result(),step=discriminator_optimizer.iterations)
                avg_g_loss.reset_states()
                avg_d_loss_real.reset_states()
                avg_d_loss_fake.reset_states()
            g_losses.append(g_loss)
            d_losses.append(d_loss_fake+d_loss_real)
            print("G_loss={}   D_loss={}".format(g_loss,d_loss_fake+d_loss_real),end='\r')
        print("\n")
        #tf.summary.trace_export(name='train_func',step=0,profiler_outdir=os.path.join(settings.config.get('paths','tb_logs'),settings.config.get('paths','log_tag')))
        for example_input, example_target in tds.take(1):
            image = generate_images(generator, example_input, example_target,epoch)
            tf.summary.image("result_epoch{:04d}".format(epoch),tf.expand_dims(image,0),step=epoch)
        
        tf.summary.scalar("epoch_time",time.time()-start,step=epoch)
        epochlist.append(epoch)
        g_loss_mean.append(np.array(g_losses).mean())
        d_loss_mean.append(np.array(d_losses).mean())
        #generate_plots(g_loss_mean,d_loss_mean,epochlist)
        
        if (epoch + 1) % 10 == 0:
            savepath = manager.save()
            #settings.config.set('training','start_epoch',str(epoch-1))
            #settings.config.write('colorize.cfg')
            print("Saved checkpoint on path: {}".format(savepath))
        
        print('Epoch[{}/{}] - {}s - G:{}  D:{}'.format(epoch+1,epochs,time.time()-start,g_loss_mean[epoch],d_loss_mean[epoch]))

