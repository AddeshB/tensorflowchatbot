from trainingprep import encoder_tokens,decoder_tokens, encoder_input_matrix,decoder_input_matrix,decoder_target_matrix
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#setting up encoder  input and LSTM layers
encoder_input_layer=Input(shape=(None,encoder_tokens))
encoder_lstm_layer=LSTM(256, return_state=True)

encoder_output_matrix, ehidden_state,ecell_state=encoder_lstm_layer(encoder_input_layer)
encoder_training_states=[ehidden_state,ecell_state]

decoder_input_layer=Input(shape=(None,decoder_tokens))
decoder_lstm_layer=LSTM(256,return_state=True, return_sequences=True)
decoder_output_matrix, dhidden_state, dcell_state=decoder_lstm_layer(decoder_input_layer, initial_state=encoder_training_states)
decoder_training_states=[dhidden_state,dcell_state]

decoder_dense_layer=Dense(decoder_tokens,activation='softmax')
decoder_output_matrix=decoder_dense_layer(decoder_output_matrix)


#set-up to train the model
tr_model=Model([encoder_input_layer,decoder_input_layer],decoder_output_matrix)
tr_model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
batch_size=10
epochs=250

tr_model.fit([encoder_input_matrix,decoder_input_matrix],decoder_target_matrix,batch_size=batch_size,epochs=epochs,validation_split=0.2)
tr_model.save('trained_model.h5')
