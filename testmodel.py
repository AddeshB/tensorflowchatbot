from trainingmodel import encoder_input_layer,decoder_input_layer, encoder_training_states,decoder_dense_layer,decoder_lstm_layer
from trainingprep import decoder_input_matrix,decoder_target_matrix, encoder_input_matrix, target_features, reverse_target_features
from preparingdata import maxseq_decoder, input_list, target_list,target_token_set,encoder_tokens,decoder_tokens

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model

import numpy as np

#since we are using a pre-trained model, we can use load_model to reload it for
#the encoder layers that are already trained

#here, we are reloading the trained model for the encoder layer
trained_model=load_model('trained_model.h5')
encoder_inputs_test=trained_model.input[0]
encoder_outputs_test,estate_hidden,estate_cell=trained_model.layers[2].output
encoder_test_states=[estate_hidden,estate_cell]
encoder_test_model=Model(encoder_inputs_test,encoder_test_states)

#unlike the trained decoder model, here we need a decoder that can decode word for word
l_dim=256
decoder_input_hidden=Input(shape=(l_dim,))
decoder_input_cell=Input(shape=(l_dim,))
decoder_input_states=[decoder_input_hidden,decoder_input_cell]
decoder_outputs,dhidden,dcell=decoder_lstm_layer(decoder_input_layer,initial_state=decoder_input_states)
decoder_states_test=[dhidden,dcell]
decoder_outputs=decoder_dense_layer(decoder_outputs)
decoder_test_model=Model([decoder_input_layer]+decoder_input_states,[decoder_outputs]+decoder_states_test)
