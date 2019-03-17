from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Input
import keras.backend as K
from vgg import VGG19, preprocess_input
from decoder import decoder_layers
from unpooling import *

LAMBDA=1

def l2_loss(x):
    return K.sum(K.square(x)) / 2

class EncoderDecoder:
    def __init__(self, input_shape=(256, 256, 3), target_layer=5,
                 decoder_path=None):
        self.input_shape = input_shape
        self.target_layer = target_layer

        self.encoder = VGG19(input_shape=input_shape, target_layer=target_layer)
        
        if decoder_path:
            self.decoder = load_model(decoder_path,custom_objects={'Unpooling':Unpooling})
        else:
            self.decoder = self.create_decoder(target_layer)

        self.model = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))


        self.loss = self.create_loss_fn(self.encoder)

        self.model.compile('adam', self.loss)

    def create_loss_fn(self, encoder):
        def get_encodings(inputs):
            encoder = VGG19(inputs, self.input_shape, self.target_layer)
            return encoder.output[0]

        def loss(img_in, img_out):
            encoding_in = get_encodings(img_in)
            encoding_out = get_encodings(img_out)
            return l2_loss(img_out - img_in) + \
                   LAMBDA*l2_loss(encoding_out - encoding_in)
        return loss

    def create_decoder(self, target_layer):
        inputs = []
        for output in self.encoder.outputs:
            inputs.append(Input(shape=[int(a) for a in output.shape[1:]]))
            print('    ',inputs[-1].shape)
        layers = decoder_layers(inputs, target_layer) #,self.masks)
        output = Conv2D(3, (3, 3), activation='relu', padding='same',
                        name='decoder_out')(layers)
        model = Model(inputs, output, name='decoder_%s' % target_layer)
        print(model.summary())
        return model

    def export_decoder(self):
        self.decoder.save('decoder_%s.h5' % self.target_layer)
