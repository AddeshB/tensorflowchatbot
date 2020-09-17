import numpy as np
import re
#importing the discord.py library
import discord
from discord.ext import commands

#these libraries are for the specifically trained seq2seq model
from testmodel import encoder_test_model,decoder_test_model
from trainingprep import input_features,target_features,reverse_target_features
from preparingdata import encoder_tokens,decoder_tokens, maxseq_decoder,maxseq_encoder

client= commands.Bot(command_prefix='?')

#the following function accepts the user input message and converts it into a matrix for the model to understand
@client.event
async def on_message(message):
  negative_responses=["stop", "exit", "quit", "bye", "goodbye"]
  def input_to_matrix(response):
    temp_tokens=re.findall(r"[\w']+|[^\s\w]", response)
    response_matrix=np.zeros((1,maxseq_encoder,encoder_tokens),dtype='float32')

    for timestep,token in enumerate(temp_tokens):
        if token in input_features:
            response_matrix[0,timestep,input_features[token]]=1.
    return response_matrix
  def chat(string_input):
    final_matrix=input_to_matrix(string_input)
    final_states=encoder_test_model.predict(final_matrix)
    output_matrix=np.zeros((1,1,decoder_tokens))
    output_matrix[0,0,target_features['<PAD>']]=1.


    string_response= ''

    end_decoding=False

    while not end_decoding:
        output_final_tokens,state_hidden,state_cell=decoder_test_model.predict([output_matrix]+final_states)

        token_sample_index=np.argmax(output_final_tokens[0,-1,:])
        token_sample_string=reverse_target_features[token_sample_index]
        string_response+=" " + token_sample_string

        if (token_sample_string=='<DAP>' or len(string_response)>maxseq_decoder):
            end_decoding=True

        output_matrix=np.zeros((1,1,decoder_tokens))
        output_matrix[0,0,token_sample_index]=1.
        final_states=[state_hidden,state_cell]
    string_response=string_response.replace("<PAD>","").replace("<DAP>","")
    return string_response
#checks if the message sent was from the user who used the command
  if message.author==client.user:
      return
  if message.content.startswith('$hello'):
      await message.channel.send("What is your name?")
      name=await client.wait_for('message')
      name_string=name.content
      await message.channel.send(f"Hi {name_string}-senpai! It's your lovely kouhai, BB!")
      reply_temp=await client.wait_for('message')
      reply_string=reply_temp.content
      while reply_string not in negative_responses:
        await message.channel.send(chat(reply_string))
        reply= await client.wait_for('message')
        reply_string=reply.content
      if reply_string in negative_responses:
          await message.channel.send("Are you leaving already senpai? We'll be talking again very soon.")
          return

#to ouput on the Terminal if the bot is ready
@client.event
async def on_ready():
  print('Bot is ready')
#test function to ensure the bot is connected to the Discord server
@client.command()
async def ping(ctx):
  await ctx.send("Pong!")
client.run('[insert_token_here]')
