from tensorflow import keras
import numpy as np
import re
from preparingdata import input_list, target_list, input_tokens,target_tokens, encoder_tokens,decoder_tokens,maxseq_decoder,maxseq_encoder

#preparing features dictionaries for both input  and target numpy matrices
input_features=dict([(token,i) for i, token in enumerate(input_tokens)])
target_features=dict([(token,i) for i, token in enumerate(target_tokens)])

#preparing reverse features dictionaries for later conversion
reverse_input_features=dict([(i,token) for token,i in input_features.items()])
reverse_target_features=dict([(i,token) for token,i in target_features.items()])

#initialzing Numpy matrixes for one-hot vectors later on
encoder_input_matrix=np.zeros((len(input_list),maxseq_encoder,encoder_tokens),dtype='float32')
decoder_input_matrix=np.zeros((len(target_list), maxseq_decoder, decoder_tokens), dtype='float32')
decoder_target_matrix=np.zeros((len(target_list), maxseq_decoder, decoder_tokens), dtype='float32')

# completing the numpy matrices to hold one-hot vectors of the features dicts
for seq_num, (input_seq,target_seq) in enumerate(zip(input_list,target_list)):
    for timestep,token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_seq)):
        encoder_input_matrix[seq_num,timestep,input_features[token]]=1.
    for timestep,token in enumerate(target_seq.split()):
        decoder_input_matrix[seq_num,timestep,target_features[token]]=1.

        #for the target matrix for teacher forcing
        if timestep>0:
            decoder_target_matrix[seq_num,timestep-1,target_features[token]]=1.
