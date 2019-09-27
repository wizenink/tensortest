import tensorflow as tf
from models import *
from metrics import *
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
    title = ['Ground Truth','Predicted']
    x = 0
    for i in range(2):
        for j in range(4):
            print("adding image to subplot: ",((j+i*4)+1))
            plt.subplot(4, 2,(j+i*4)+1)
            plt.title(title[x%2])
            # getting the pixel values between [0, 1] to plot it.
            if(len(display_list[x].shape) == 2):
                plt.imshow(display_list[x],cmap='gray')
            else:
                plt.imshow(display_list[x])
            plt.axis('off')
            x = x+1
    plt.savefig(settings.config['paths']['results']+'image_at_epoch_{:04d}.png'.format(epoch))
    #cv2.imwrite('result_{:04d}.png'.format(epoch),cv2.cvtColor(gen_output,cv2.COLOR_YUV2BGR))
    
    #plt.show()

def generate_plots(g_loss_mean,d_loss_mean,epochs):
    plt.figure()
    plt.plot(g_loss_mean,epochs,'g',label='Generator loss')
    plt.plot(d_loss_mean,epochs,'b',label='Discriminator loss')
    plt.savefig(settings.config['paths']['plots']+'losses.png')
@tf.function
def train_step(input_image,target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print("SHAPE:",input_image.shape)
        gen_output = generator(input_image,training=True)

        disc_real = discriminator([input_image,target],training=True)
        disc_gen = discriminator([input_image,gen_output],training=True)
        g_loss = gen_loss(disc_gen,gen_output,target)
        d_loss = disc_loss(disc_real,disc_gen)

        generator_gradients = gen_tape.gradient(g_loss,generator.trainable_variables) 

        discriminator_gradients = disc_tape.gradient(d_loss,discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
        return g_loss,d_loss

def fit(ds,epochs):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
    
    g_loss_mean = []
    d_loss_mean = []
    epochlist = []
    for epoch in range(epochs):
        start = time.time()
        g_losses = []
        d_losses = []
        for input_image,target in ds:
            g_loss,d_loss = train_step(input_image,target)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            print("G_loss={}   D_loss={}".format(g_loss,d_loss),end='\r')
        print("\n")

        for example_input, example_target in ds.take(1):
            generate_images(generator, example_input, example_target,epoch)


        epochlist.append(epoch)
        g_loss_mean.append(np.array(g_losses).mean())
        d_loss_mean.append(np.array(d_losses).mean())
        generate_plots(g_loss_mean,d_loss_mean,epochlist)
        
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print('Epoch[{}/{}] - {}s - G:{}  D:{}'.format(epoch+1,epochs,time.time()-start,g_loss_mean[epoch],d_loss_mean[epoch]))

