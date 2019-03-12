from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Input
from keras.utils.data_utils import get_file
import keras.backend as K
import h5py
import numpy as np
import tensorflow as tf
from unpooling import *

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

WEIGHTS_PATH = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models',
                        file_hash='253f8cb515780f3b799900260a226db6')

def vgg_layers(inputs, target_layer):
    masks = [] 
    # Block 1
    #(256,256,3)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs) #(256,256,64)
    if target_layer == 1:
        return [x,*masks]
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x) #(256,256,64) 
    mask_1 = MaxPoolingMask2D(pool_size=(2,2),strides=(2, 2),name='block1_pool_index')(x) #(256,256,64)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) #(128,128,64)
    masks.append(mask_1)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x) #(128,128,128)
    if target_layer == 2:
        return [x,*masks]
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    mask_2 = MaxPoolingMask2D(pool_size=(2,2), strides=(2, 2), name='block2_pool_index')(x) #(64,64,128)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    masks.append(mask_2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x) #(64,64,256)
    if target_layer == 3:
        return [x,*masks]
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    mask_3 = MaxPoolingMask2D(pool_size=(2,2), strides=(2, 2), name='block3_pool_index')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x) #(32,32,256)
    masks.append(mask_3)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x) #(32,32,512)
    if target_layer == 4:
        return [x,*masks]
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    mask_4 = MaxPoolingMask2D(pool_size=(2,2), strides=(2, 2), name='block4_pool_index')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) #(16,16,512)
    masks.append(mask_4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x) #(16,16,512)
    return [x,*masks]


def load_weights(model):
    f = h5py.File(WEIGHTS_PATH)
    layer_names = [name for name in f.attrs['layer_names']]

    for layer in model.layers:
        b_name = layer.name.encode()
        if b_name in layer_names:
            g = f[b_name]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)
            layer.trainable = False

    f.close()


def VGG19(input_tensor=None, input_shape=None, target_layer=1):
    """
    VGG19, up to the target layer (1 for relu1_1, 2 for relu2_1, etc.)
    """
    if input_tensor is None:
        inputs = Input(shape=input_shape, name='vgg-input-0')
    else:
        inputs = Input(tensor=input_tensor, shape=input_shape, name='vgg-input-1')
    model = Model(inputs, vgg_layers(inputs, target_layer), name='vgg19')
    load_weights(model)
    return model
    #return (model,vgg_layers(inputs, target_layer)[1])


def preprocess_input(x):
    # Convert 'RGB' -> 'BGR'
    if type(x) is np.ndarray:
        x = x[..., ::-1]
    else:
        x = tf.reverse(x, [-1])

    return x - MEAN_PIXEL
