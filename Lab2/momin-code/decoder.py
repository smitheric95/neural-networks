from keras.layers import Conv2D, UpSampling2D, Input
from unpooling import *
from keras.layers import Lambda


def decoder_layers(inputs, layer):#, masks):
    print('layer',layer)
    inp = inputs[0]
    masks = inputs[1:]
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(inp)
    if layer == 1:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block4_upsample')(x)
    x = Unpooling()([x, masks[-1]])
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv4')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    if layer == 2:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block3_upsample')(x)
    x = Unpooling()([x, masks[-2]]) #FLAG this is the problem
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    if layer == 3:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block2_upsample')(x)
    x = Unpooling()([x, masks[-3]])
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block2_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    if layer == 4:
        return x
    
    #x = UpSampling2D((2, 2), name='decoder_block1_upsample')(x)
    x = Unpooling()([x, masks[-4]])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block1_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block1_conv1')(x)
    if layer == 5:
        return x

