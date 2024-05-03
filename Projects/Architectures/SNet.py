
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf

########################################################################################################################
# UNet
########################################################################################################################

class SNet:

    ########################################################################################################################
    # downsample
    ########################################################################################################################

    def downsample(self,filters, size, apply_batchnorm=True,strides=2):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    ########################################################################################################################
    # upsample
    ########################################################################################################################

    def upsample(self,filters, size, apply_dropout=False, strides=2):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    ########################################################################################################################
    # __init__
    ########################################################################################################################

    def __init__(self, inputChannels, outputChannels, inputResolutionU, inputResolutionV, lastActivation= 'tanh'):

        self.outputChannels = outputChannels

        inputs = tf.keras.layers.Input(shape=[inputResolutionV,inputResolutionU,inputChannels]) #(bs, 1024, 1024, 15)

        down_stack = [
        self.downsample(64, 3, apply_batchnorm=False), # (bs, 512, 512, 16)
        self.downsample(64, 3,strides=1), # (bs, 512, 512, 16)
        self.downsample(128, 3), # (bs, 256, 256, 32)
        self.downsample(128, 3,strides=1), # (bs, 512, 512, 16)
        ]

        up_stack = [
        self.upsample(128, 3) ,# (bs, 128, 128, 64)
        self.upsample(64, 3,strides=1),
        self.upsample(64, 3),  # (bs, 256, 256, 32)
        self.upsample(32, 3,strides=1),
        self.upsample(16, 3),  # (bs, 512, 512, 16)
        self.upsample(16, 3,strides=1),
        self.upsample(16, 3),  # (bs, 512, 512, 16)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            #skips.append(x)
        

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up in up_stack:

            x = up(x)
            #x = tf.keras.layers.Concatenate()([x, skip])
        end = tf.keras.layers.Conv2DTranspose(self.outputChannels,
                                             3,
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation= lastActivation) # (bs, 1024, 1024, 3)

        x = end(x)# (bs, 1024, 1024, 3)

        ##################################

        if lastActivation == 'tanh':
            x = (x + 1.0) / 2.0
        else:
            x = tf.keras.layers.ReLU()(x)

        self.model= tf.keras.Model(inputs=inputs, outputs=x)

        print(self.model.summary())

    ########################################################################################################################
    # Additional backbone
    ########################################################################################################################

    def initTinyBackbone(self, inputResV, inputResU, inputChannels, outputChannels):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[inputResV, inputResU, inputChannels], name='input_image')

        x= tf.keras.layers.Conv2D(outputChannels, (1,1), strides=1, padding='same', kernel_initializer=initializer, use_bias=False, activation='tanh')(inp)
        x = (x + 1.0) / 2.0

        self.tinyBackbone = tf.keras.Model(inputs=inp, outputs=x)

        print(self.tinyBackbone.summary())

    ########################################################################################################################
    # discriminator
    ########################################################################################################################

    def initDiscriminator(self,inputResV, inputResU, inputChannels, outputChannels):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[inputResV, inputResU, inputChannels], name='input_image')
        tar = tf.keras.layers.Input(shape=[inputResV, inputResU, outputChannels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 1024, 1024, channels*2)

        down1 = self.downsample(16, 8, False)(x)  # (batch_size, 512, 512, 16)
        down2 = self.downsample(32, 4)(down1)  # (batch_size, 256, 256, 32)
        down3 = self.downsample(64, 4)(down2)  # (batch_size, 128, 128, 64)
        down4 = self.downsample(128, 4)(down3)  # (batch_size, 64, 64, 128)
        down5 = self.downsample(256, 4)(down4)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,  kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        self.discriminator =  tf.keras.Model(inputs=[inp, tar], outputs=last)

        print(self.discriminator.summary())

    ########################################################################################################################
    # Discriminator loss
    ########################################################################################################################

    def discriminator_loss (self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss) * 0.0
        return total_disc_loss

    ########################################################################################################################
    # Generator loss
    ########################################################################################################################

    def generator_loss(self, disc_generated_output, gen_output, target):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        gan_loss = tf.reduce_mean(gan_loss) * 0.0

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) * 1.0
        total_gen_loss = gan_loss + l1_loss

        return total_gen_loss, gan_loss, l1_loss
if __name__=="__main__":
    SRNet = SNet(75, 3, 1028,752)
    # import time
    # for i in range(10):

    #     temp=tf.zeros((1,940,1285,75))
    #     a=time.time()
    #     c=SRNet.model(temp)
    #     b=time.time()
    #     print(a-b)