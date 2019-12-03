import tensorflow as tf
import os
import settings

LAMBDA = 50
LAMBDA2 = 50

loss_metric = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.2)

#def loss_metric(y_true, y_pred):
#    return tf.reduce_mean(y_true * y_pred) 

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'cp')

# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=,discriminator=)


def disc_loss(y_true,y_pred):
    real_loss = loss_metric(tf.zeros_like(y_true),y_true)

    #Perform 1-sided label smoothing
    #labels = (0.8-1.0) * tf.random.uniform(y_true.shape) + 1.0
    #real_loss = loss_metric(labels,y_true)


    pred_loss = loss_metric(tf.ones_like(y_pred),y_pred)

    total_loss = real_loss + pred_loss
    
    return pred_loss,real_loss

def gen_loss(disc_pred,gen_pred,target):

    gan_loss = loss_metric(tf.zeros_like(disc_pred),disc_pred)

    l1_loss = tf.reduce_mean(tf.abs(target-gen_pred))
    l2_loss = tf.reduce_mean(tf.square(target-gen_pred))

    total_loss = gan_loss + (LAMBDA * l1_loss + LAMBDA2 * l2_loss) 

    return total_loss
