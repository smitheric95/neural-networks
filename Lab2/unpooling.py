from keras import backend as K
from keras.layers.convolutional import UpSampling2D
from keras.layers import MaxPooling2D, Layer
import tensorflow as tf

class MaxPoolingMask2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, **kwargs):
        super(MaxPoolingMask2D, self).__init__(pool_size, strides, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
        pooled = K.pool2d(inputs, pool_size, strides, pool_mode='max')
        upsampled = UpSampling2D(size=pool_size)(pooled)
        indexMask = K.tf.equal(inputs, upsampled)
        assert indexMask.get_shape().as_list() == inputs.get_shape().as_list()
        return indexMask

    def get_output_shape_for(self, input_shape):
        return input_shape

class Unpooling(Layer):
    def __init__(self, **kwargs):
        super(Unpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        layer = x[0]
        layer_mask = x[1]
        print('unpooling input - layer shape:',layer.shape,'mask shape:',layer_mask.shape)
        mask_shape = layer_mask.get_shape().as_list()
        layer_shape = layer.get_shape().as_list()
        pool_size = (2,2) #(mask_shape[1] / layer_shape[1], mask_shape[2] / layer_shape[2])
        #on_success = UpSampling2D(size=pool_size)(layer)
        #on_success = K.resize_images(layer, 2, 2, 'channels_last', 'nearest')

        on_success = tf.keras.backend.resize_images(layer, 2, 2, 'channels_last')

        #while on_success.shape[1] < mask_shape[1]:
            #pool_size = (pool_size[0]*2, pool_size[1]*2)
            #on_success = UpSampling2D(size=pool_size)(layer)

        #while on_success.shape[-1] > mask_shape[-1]:
            #on_success = on_success[:,:,:,::2]
        on_fail = K.zeros_like(on_success)
        output =  K.tf.where(K.tf.cast(layer_mask,bool), on_success, on_fail)
        print('    ',output.shape)
        return output


    def compute_output_shape(self, input_shape):
        return input_shape[1]
