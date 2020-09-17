from tensorflow import keras
from itertools import zip_longest
import re

file="sample.txt"
#processes data from raw file
with open(file, 'r', encoding='utf-8') as f:
    raw_lines=f.read().split('\n')
#un comment the underneath line for the discord demo bot
#  cleaned_lines=[re.sub(">","",line) for line in raw_lines]
#grouper function from Official Python documentation for iter-tools
def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
raw_pairs=list(grouper(2,raw_lines))
#creates list of vocab words to find number of each in the future
input_list=[]
target_list=[]
#creates list of vocab tokens for future dictionaries
input_token_set=set()
target_token_set=set()
#see if you can index slice so that it only goes through like 10k lines
for line in raw_pairs[:500]:
      input_line,target_line=line[0],line[1]

      input_list.append(input_line)

      target_line=" ".join(re.findall(r"[\w']+|[^\s\w]",target_line))
      target_line= '<PAD> ' + target_line + ' <DAP>'
      target_list.append(target_line)

      for token in re.findall(r"[\w']+|[^\s\w]",input_line):
          if token not in input_token_set:
              input_token_set.add(token)
      for token in target_line.split():
          if token not in target_token_set:
              target_token_set.add(token)
input_tokens=sorted(list(input_token_set))
target_tokens=sorted(list(target_token_set))

encoder_tokens=len(input_tokens)
decoder_tokens=len(target_tokens)

maxseq_encoder=max([len(re.findall(r"[\w']+|[^\s\w]", line)) for line in input_list])
maxseq_decoder=max([len(re.findall(r"[\w']+|[^\s\w]", line)) for line in target_list])
