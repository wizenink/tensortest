import tensorflow as tf
from models import *
from metrics import *
import time
EPOCHS = 100

@tf.function
def train_step(input_image,target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image,training=True)

        disc_real = discriminator([input_image,target],training=True)
        disc_gen = discriminator([input_image,gen_output],training=True)
        g_loss = gen_loss(disc_gen,gen_output,target)
        d_loss = disc_loss(disc_real,disc_gen)

        generator_gradients = gen_tape.gradient(g_loss,generator.trainable_variables) 

        discriminator_gradients = disc_tape.gradient(d_loss,discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

def fit(ds,epochs):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
    for epoch in range(epochs):
        start = time.time()

        for input_image,target in ds:
            train_step(input_image,target)
        
        print('NEH')
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print('Epoch[{}/{}] - {}'.format(epoch+1,epochs,time.time()-start))

