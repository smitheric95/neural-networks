from keras.layers import Conv2D, UpSampling2D, Input
from unpooling import *
from keras.layers import Lambda


def decoder_layers(inputs, layer):#, masks):
    layer = int(layer)
    print('layer',layer)
    inp = inputs[0]
    masks = inputs[1:]
    num_filters = int(inputs[-1].shape[-1])
    #(256,256,3)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(inp) #(16,16,512)
    if layer == 1:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block4_upsample')(x)
    print('Doing first unpooling')
    x = Unpooling(name='first-unpool')([x, masks[-1]])  #(32,32,512)
    num_filters = int(inputs[-2].shape[-1])
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block4_conv4')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block4_conv3')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    if layer == 2:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block3_upsample')(x)
    print('Doing second unpooling')
    x = Unpooling(name='second-unpool')([x, masks[-2]]) #FLAG this is the problem #(64,64,512)
    num_filters = int(inputs[-3].shape[-1])
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block3_conv4')(x) #(64,64,256)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    if layer == 3:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block2_upsample')(x)
    print('Doing third unpooling')
    x = Unpooling(name='second-to-last-unpool')([x, masks[-3]]) #(128,128,256)
    num_filters = int(inputs[-4].shape[-1])
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block2_conv2')(x) #(128,128,128)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    if layer == 4:
        return x
    
    #x = UpSampling2D((2, 2), name='decoder_block1_upsample')(x)
    print('Doing fourth unpooling')
    x = Unpooling(name='last-unpool')([x, masks[-4]]) #(256,256,128)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block1_conv2')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', name='decoder_block1_conv1')(x) #(256,256,64)
    if layer == 5:
        return x

