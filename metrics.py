import tensorflow as tf
import os


LAMBDA = 300

loss_metric = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-3,beta_1=0.5)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'cp')

# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=,discriminator=)


def disc_loss(y_true,y_pred):
    real_loss = loss_metric(tf.ones_like(y_true),y_true)
    pred_loss = loss_metric(tf.zeros_like(y_pred),y_pred)

    total_loss = real_loss + pred_loss
    
    return total_loss

def gen_loss(disc_pred,gen_pred,target):

    gan_loss = loss_metric(tf.ones_like(disc_pred),disc_pred)

    l1_loss = tf.reduce_mean(tf.abs(target-gen_pred))

    total_loss = gan_loss + (LAMBDA * l1_loss)

    return total_loss
