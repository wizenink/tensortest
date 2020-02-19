import tensorflow as tf
from attention import SelfAttnModel
from spectral import SpectralConv2D
import pix2pix
OUTPUT_CHANNELS = 2

def downsample(filters,size,batchnorm = True):
    init = tf.random_normal_initializer(0.,0.02)

    result = tf.keras.Sequential()
    result.add(SpectralConv2D(filters,size,strides=2,padding='same',use_bias=False))

    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters,size,dropout = False):
    init = tf.random_normal_initializer(0.,0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same',use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
        '''
        conditioning = Input(shape=(None,None,self.G_INPUT_CHANNELS))
        filters = 32
        c1 = self.addConvStep(input_layer,filters,conditioning)
        filters = 32
        c2 = self.addConvStep(c1,filters,conditioning)
        filters = 16
        c3 = self.addConvStep(c2,filters,conditioning)
        filters = 2
        c4 = SeparableConv2D(filters,self.KERNEL_SIZE,padding='same',activation=customAct)(c3)
        model = Model([input_layer,conditioning],c4)
        print("--Generator--")
        model.summary()
        '''
        init = tf.random_normal_initializer(0.,0.02)
        noise = tf.keras.layers.Input(shape=(256, 256, 1))
        conditioning = Input(shape=(256,256,1))
        #hid,att2 = SelfAttnModel(1)(conditioning)
        #noise = tf.keras.layers.Input(shape=(None,None,1))
        hid = tf.keras.layers.Concatenate()([noise,conditioning])
        hid = SpectralConv2D(2, (3, 3), activation='relu', padding='same')(conditioning)
        hid = tf.keras.layers.BatchNormalization()(hid)
        hid = SpectralConv2D(16, (3, 3), activation='relu', padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization()(hid)
        hid = SpectralConv2D(16, (3, 3), activation='relu', padding='same')(hid)
        hid = SpectralConv2D(16, (3, 3), activation='relu', padding='same')(hid)
        #hid,att1 = SelfAttnModel(16)(hid)
        hid = SpectralConv2D(2, (3, 3)  ,activation='relu', padding='same')(hid)
        hid =  SpectralConv2D(2, (3, 3), activation='tanh',padding='same')(hid)
        model = tf.keras.Model(conditioning,hid)
        
        return model




def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 2], name='input_image')
  inpn = tf.keras.layers.GaussianNoise(0.1)(inp)
  tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')

  x = tf.keras.layers.concatenate([inpn, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  #down2,att1 = SelfAttnModel(128)(down2)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
  
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)


  return tf.keras.Model(inputs=[inp, tar], outputs=last)


#generator = Generator()
#discriminator = Discriminator()
generator = pix2pix.unet_generator(2,norm_type='instancenorm')
discriminator = pix2pix.discriminator(norm_type='instancenorm',target=True)