import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import random

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/place'

!ls

from torch.utils.data import Dataset


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1
        
        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2   

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx] #data_list[idx]

    def __len__(self):
        return len(self.data) # len(data_list)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()
        
        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens
                    
        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                for token in tokens:
                    seq.append(src_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")
            
        return data_list, src_vocab, tgt_vocab, src_max, tgt_max

# !mkdir numbers__place_value

# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...

#DATASET_DIR = "/content/drive/My Drive/BI/"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = "numbers__place_value"
# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

#src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"


train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

#src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False)

from torch.utils.data import DataLoader

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

src_vocab.id_to_string

tgt_vocab.id_to_string

# Example batch
batch = next(iter(train_data_loader))

source = batch[0]  # source sequence
print(source.shape)
target = batch[1]  # target sequence
print(target.shape)

# example source/target pair
example_source_sequence = []

for i in source[0]:
    example_source_sequence.append(src_vocab.id_to_string[i.item()])
print(source[0])
print(example_source_sequence)

print(''.join(example_source_sequence))

example_target_sequence = []

for i in target[0]:
    example_target_sequence.append(tgt_vocab.id_to_string[i.item()])

print(example_target_sequence)

########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

"""3.Model"""

import torch.nn.functional as F
from torch.utils.data import dataset

class TransformerModel(nn.Module):

  def __init__(self, etoken, dtoken, d_model, nhead, d_hid, n_encoder_layers, n_decoder_layers, pad_ID,device, dropout = 0.0):
    super().__init__()
    self.pos_encoder = PositionalEncoding(d_model , dropout)
    self.transformer = nn.Transformer(d_model, nhead, n_encoder_layers, 
                                     n_decoder_layers, d_hid, dropout)
    self.encoder_embedder = nn.Embedding(etoken, d_model)
    self.decoder_embedder = nn.Embedding(dtoken, d_model)
    self.d_model = d_model
    self.pad_ID = pad_ID
    self.classifier = nn.Linear(d_model, dtoken)

  def forward(self, src, tgt):
    src_embed = self.encoder_embedder(src) 
    src_embed = self.pos_encoder(src_embed)
    tgt_embed = self.decoder_embedder(tgt) 
    tgt_embed = self.pos_encoder(tgt_embed)
    
    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
    src_key_padding_mask = self.pad_mask(src)
    tgt_key_padding_mask = self.pad_mask(tgt)
    memory_key_padding_mask = self.pad_mask(src)
    output = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask, 
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
    classified = self.classifier(output)
    return classified

  def pad_mask(self, tensor):
    return (tensor == self.pad_ID).transpose(0,1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""4.Greedy Search"""

# 5. Accuracy computation and returns example q and a.
def print_prediction(predictions,target,vocab):
  #predictions:(Seq Length,Batch Size)
  #target:(Seq Length,Batch Size)
  start_pred = predictions.T[-1][0]
  start_targ = target.T[-1][0]
  
  predictionsT = predictions[1:].T
  targetT = target[1:].T
  correct = 0
  
 
  for i,j in enumerate(predictionsT):
    breaker = False
    for ind,item in enumerate(j):
      if item==targetT[i][ind]: 
        do_nothing = 0
      elif targetT[i][ind]==vocab.pad_id:
        do_nothing = 0
      else:
        breaker = True
        break
    if breaker==False:
      correct += 1
     
  v_acc = correct/(i+1)

  answer = ""
  for word_id in j:
    answer += str(vocab.id_to_string[word_id.item()]) +" "
  pred = answer
  pred = str(vocab.id_to_string[start_pred.item()])+" " + pred
  answer2 = ""
  for word_id in targetT[i]:
    answer2 += str(vocab.id_to_string[word_id.item()]) +" "
  actual = answer2
  actual = str(vocab.id_to_string[start_targ.item()])+" " + actual
  return (v_acc,pred,actual)

#this
def greedy_search(model, input, target, vocab, device,loss):
  # input: (Batch Size , Seq Length)
  # target: (Batch Size , Seq Length)
  src = input.T.to(device) #(Seq Length,Batch Size)
  tgt = target.T.to(device) #(Seq Length,Batch Size)
 
  src_embed = model.encoder_embedder(src) 
  src_embed = model.pos_encoder(src_embed)

  sos_batch = torch.full((1,tgt.shape[1]),vocab.sos_id,dtype=torch.float64).to(device)
  eos_batch = torch.full((1,tgt.shape[1]),vocab.eos_id,dtype=torch.float64).to(device)
  
  memory = model.transformer.encoder(src_embed, src_key_padding_mask = model.pad_mask(src))

  pred_id = torch.clone(sos_batch)
  eos_id = torch.clone(eos_batch)
  dec_input = torch.clone(sos_batch)

  tgt_mask = None
  S = nn.Softmax(dim=2)
  memory_key_padding_mask = model.pad_mask(src)
  
  while(not torch.all(pred_id == eos_id) and dec_input.shape[0]<tgt.shape[0]):
    
    dec_input = dec_input.type(torch.int)
    dec_input = dec_input.to(device)

    tgt_embed = model.decoder_embedder(dec_input) 
    tgt_embed = model.pos_encoder(tgt_embed)

    output = model.transformer.decoder(tgt_embed, memory, tgt_mask=tgt_mask, 
                              tgt_key_padding_mask=model.pad_mask(dec_input),
                              memory_key_padding_mask=memory_key_padding_mask)
    output = model.classifier(output) 
  
    scores = S(output)
    pred_id = scores.max(2)[1]
    
    pred_id = pred_id[-1].unsqueeze(0)
    pred_id = pred_id.type(torch.int)
    dec_input = torch.cat([dec_input, pred_id])         
    tgt_mask = model.transformer.generate_square_subsequent_mask(dec_input.shape[0]).to(device)
    
  
  
  if output.shape[0] != tgt[1:].shape[0]:
    return (None,(None,None,None))


  L = loss(output.flatten(0,1) , tgt[1:].flatten(0,1))
  
  
  return (L,print_prediction(dec_input,tgt,vocab))

num_encoders = 3
num_decoders = 2
batch_size = 64
d_model = 256
h_dim = 1024
heads = 8
gradient_clip = 0.1

model = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

def train(model, loss, train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,
          tr_loss, val_loss, tr_acc, val_acc, 
          epochs=1, update_option=10, print_option=157, 
          save=None, save_mod=None, names=None, algebra=None):
  
  S = nn.Softmax(dim=2)
  for epoch in range(epochs):
    running_loss = 0
    counter = 0
    correct = 0
    counter_acc = 0
    
    for i,j in enumerate(iter(train_data_loader)):

      input = j[0].T.to(device)
      target_in = j[1][:,:-1].T.to(device)
      target_out = j[1][:,1:].T.to(device)
      
      output = model.forward(input,target_in)
     
      L = loss(output.flatten(0,1) , target_out.flatten(0,1))
      running_loss += torch.exp(L)
      counter += 1
      
      
      scores = S(output)
      pred_id = scores.max(2)[1]
      predictionsT = pred_id.T
      targetT = target_out.T

      for ind,j in enumerate(predictionsT):
        breaker = False
        for index,item in enumerate(j):
          if item==targetT[ind][index]: 
            do_nothing = 0
          elif targetT[ind][index]==tgt_vocab.pad_id:
            do_nothing = 0
          else:
            breaker = True
            break
        if breaker == False:
          correct+=1
      counter_acc += ind+1
      
      
      L.backward()

      if save!=None and algebra==None:
        if i%save_mod == 0 and i!=0:
          loc = "Epoch"+str(epoch)+" "+save+str(i)+".pt"
          names.append(loc)
          torch.save({
            'i': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, loc)


      if i%update_option == 0: 
        torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
        
      if i%print_option == 0: 
        average_loss = running_loss/counter
        tr_loss.append(average_loss)
        average_acc = (correct*100)/(counter_acc)
        tr_acc.append(average_acc)
        print(f"Epoch: {epoch}, chunk index:{i}, perplexity: {average_loss}")
        print(f"Epoch: {epoch}, chunk index:{i}, train accuracy: {average_acc}")
        running_loss = 0
        counter = 0
        correct = 0
        counter_acc = 0
        
        total_v_acc = 0
        validation_loss = 0
        model.eval()
        with torch.no_grad():
          rand_index = random.randint(0,len(valid_data_loader)-1)
          for i,j in enumerate(iter(valid_data_loader)):
            input = j[0].to(device) #(B,S)
            vloss,(v_acc,pred,actual) = greedy_search(model,input,j[1].to(device),tgt_vocab,device,loss)
            if i == rand_index:
              print_input = input
              print_pred = pred
              print_actual = actual
            if v_acc!= None:
              total_v_acc += v_acc
            if vloss!= None:
              validation_loss += torch.exp(vloss)
          if validation_loss == 0:
              print_vl = validation_loss
          else:
            print_vl = validation_loss.item()

          val_loss.append(print_vl/(i+1))
          val_acc.append((total_v_acc*100)/(i+1))
          
          print("v_acc:",(total_v_acc*100)/(i+1))
          print("v_loss:",print_vl/(i+1))
          question = ""
          for letter in print_input[-1]:
            if src_vocab.pad_id != letter:
              question+= str(src_vocab.id_to_string[letter.item()])
          print("Question",question)
          print("prediction",print_pred)
          print("actual",print_actual)
    if save!=None:
      loc = save+"loss"+str(epoch)+".pt"
      torch.save({
        'epoch': epoch,
        'tr_loss': tr_loss,
        'val_loss':val_loss,
        'tr_acc':tr_acc,
        'val_acc':val_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, loc)

  return (tr_loss,val_loss,tr_acc,val_acc)

tr_loss = []
val_loss = []
tr_acc = []
val_acc = []

train(model, loss, train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,
      tr_loss, val_loss, tr_acc, val_acc)

import matplotlib.pyplot as plt
plt.plot(tr_loss)
plt.plot(val_loss[1:])
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc)
plt.plot(val_acc)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

def stats(tr_loss, val_loss, tr_acc, val_acc):
  print("max_tr loss:",max(tr_loss).item())
  print("index of max_tr loss:",tr_loss.index(max(tr_loss)))
  print("min_tr loss:",min(tr_loss).item())
  print("index of min_tr loss:",tr_loss.index(min(tr_loss)))

  print("max_tr acc:",max(tr_acc))
  print("index of max_tr acc:",tr_acc.index(max(tr_acc)))
  print("min_tr acc:",min(tr_acc))
  print("index of min_tr acc:",tr_acc.index(min(tr_acc)))

  print("max_val loss:",max(val_loss[1:]))
  print("index of max_val loss:",val_loss[1:].index(max(val_loss[1:]))+1)
  print("min_val loss:",min(val_loss[1:]))
  print("index of min_val loss:",val_loss[1:].index(min(val_loss[1:]))+1)

  print("max_val acc:",max(val_acc))
  print("index of max_val acc:",val_acc.index(max(val_acc)))
  print("min_val acc:",min(val_acc))
  print("index of min_val acc:",val_acc.index(min(val_acc)))

print("max_val loss:",max(val_loss[1:]))
print("index of max_val loss:",val_loss[1:].index(max(val_loss[1:]))+1)
print("min_val loss:",min(val_loss[1:]))
print("index of min_val loss:",val_loss[1:].index(min(val_loss[1:]))+1)

print("max_val acc:",max(val_acc))
print("index of max_val acc:",val_acc.index(max(val_acc)))
print("min_val acc:",min(val_acc))
print("index of min_val acc:",val_acc.index(min(val_acc)))

print("max_tr loss:",max(tr_loss).item())
print("index of max_tr loss:",tr_loss.index(max(tr_loss)))
print("min_tr loss:",min(tr_loss).item())
print("index of min_tr loss:",tr_loss.index(min(tr_loss)))

print("max_tr acc:",max(tr_acc))
print("index of max_tr acc:",tr_acc.index(max(tr_acc)))
print("min_tr acc:",min(tr_acc))
print("index of min_tr acc:",tr_acc.index(min(tr_acc)))

num_encoders = 3
num_decoders = 1 #-1 decoder
batch_size = 64
d_model = 256
h_dim = 1024
heads = 8
gradient_clip = 0.1

model2 = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model2.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model2.parameters(),lr=0.0001)

tr_loss2 = []
val_loss2 = []
tr_acc2 = []
val_acc2 = []

train(model2,loss,train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,tr_loss2,val_loss2,tr_acc2,val_acc2)

import matplotlib.pyplot as plt
plt.plot(tr_loss2)
plt.plot(val_loss2)
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc2)
plt.plot(val_acc2)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

stats(tr_loss2, val_loss2, tr_acc2, val_acc2)

num_encoders = 1 #-1 encoder
num_decoders = 1 #-1 decoder
batch_size = 64
d_model = 256
h_dim = 1024
heads = 8
gradient_clip = 0.1

model3 = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model3.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model3.parameters(),lr=0.0001)

tr_loss3 = []
val_loss3 = []
tr_acc3 = []
val_acc3 = []

train(model3,loss,train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,tr_loss3,val_loss3,tr_acc3,val_acc3)

import matplotlib.pyplot as plt
plt.plot(tr_loss3)
plt.plot(val_loss3)
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc3)
plt.plot(val_acc3)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

stats(tr_loss3, val_loss3, tr_acc3, val_acc3)

model4 = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model4.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model4.parameters(),lr=0.001)

tr_loss4 = []
val_loss4 = []
tr_acc4 = []
val_acc4 = []

train(model4,loss,train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,tr_loss4,val_loss4,tr_acc4,val_acc4)

plt.plot(tr_loss4)
plt.plot(val_loss4)
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc4)
plt.plot(val_acc4)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

stats(tr_loss4, val_loss4, tr_acc4, val_acc4)

num_encoders = 1 #-1 encoder
num_decoders = 1 #-1 decoder
batch_size = 64
d_model = 256
h_dim = 1024
heads = 2 #2
gradient_clip = 0.1

model5 = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model5.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model5.parameters(),lr=0.001)

tr_loss5 = []
val_loss5 = []
tr_acc5 = []
val_acc5 = []

train(model5,loss,train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,tr_loss5,val_loss5,tr_acc5,val_acc5)

plt.plot(tr_loss5)
plt.plot(val_loss5)
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc5)
plt.plot(val_acc5)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

stats(tr_loss5, val_loss5, tr_acc5, val_acc5)

num_encoders = 1 #-1 encoder
num_decoders = 1 #-1 decoder
batch_size = 64
d_model = 256
h_dim = 512 #-512
heads = 8
gradient_clip = 0.1

model6 = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model6.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model6.parameters(),lr=0.001)

tr_loss6 = []
val_loss6 = []
tr_acc6 = []
val_acc6 = []

train(model6,loss,train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,tr_loss6,val_loss6,tr_acc6,val_acc6)

import matplotlib.pyplot as plt
plt.plot(tr_loss6)
plt.plot(val_loss6)
plt.xlabel('Batches in Epoch')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over batches in 1 epoch')

plt.plot(tr_acc6)
plt.plot(val_acc6)
plt.xlabel('Batches in Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over batches in 1 epoch')

stats(tr_loss6, val_loss6, tr_acc6, val_acc6)

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/sort'

# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...


TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"


# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

#src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"


train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

#src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False)

from torch.utils.data import DataLoader

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

src_vocab.id_to_string

tgt_vocab.id_to_string

# Example batch
batch = next(iter(train_data_loader))

source = batch[0]  # source sequence
print(source.shape)
target = batch[1]  # target sequence
print(target.shape)

# example source/target pair
example_source_sequence = []

for i in source[0]:
    example_source_sequence.append(src_vocab.id_to_string[i.item()])
print(source[0])
print(example_source_sequence)

print(''.join(example_source_sequence))

example_target_sequence = []

for i in target[0]:
    example_target_sequence.append(tgt_vocab.id_to_string[i.item()])

print(example_target_sequence)

num_encoders = 1
num_decoders = 1
batch_size = 64
d_model = 256
h_dim = 512
heads = 8
gradient_clip = 0.1

model = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

sort_tr_loss = []
sort_val_loss = []
sort_tr_acc = []
sort_val_acc = []

checkpoint = torch.load('sort break6.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']

train(model, loss, train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,
      sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc,
      5, 10, 157,
      save="sort2-",save_mod=10000,names=[])

loc = "sort"+" "+"break8"+".pt"
torch.save({
  'epoch': 0,
  'sort_tr_loss': sort_tr_loss,
  'sort_val_loss':sort_val_loss,
  'sort_tr_acc':sort_tr_acc,
  'sort_val_acc':sort_val_acc,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict()
  }, loc)

len(sort_val_loss)

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/saves/sort'

checkpointo = torch.load('sort break0.pt',map_location=device)
sort_tr_losso = checkpointo['sort_tr_loss']
sort_val_losso = checkpointo['sort_val_loss']
sort_tr_acco = checkpointo['sort_tr_acc']
sort_val_acco = checkpointo['sort_val_acc']

len(sort_val_losso)

checkpoint = torch.load('sort break8.pt',map_location=device)
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']

len(sort_val_loss)

len(sort_val_loss)-len(sort_val_losso)

import matplotlib.pyplot as plt
plt.plot(sort_tr_loss)
plt.plot(sort_val_loss)
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over all batches')

import matplotlib.pyplot as plt
plt.plot(sort_tr_losso)
plt.plot(sort_val_losso)
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over first 160 batches')

plt.plot(sort_tr_acco)
plt.plot(sort_val_acco)
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over first 160 batches')

stats(sort_tr_losso, sort_val_losso, sort_tr_acco, sort_val_acco)

plt.plot(sort_val_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Validation Perplexity'])
plt.title('Perplexity over next 316 batches')

plt.plot(sort_tr_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity'])
plt.title('Perplexity over next 316 batches')

import matplotlib.pyplot as plt
plt.plot(sort_tr_loss[len(sort_tr_losso):])
plt.plot(sort_val_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over next 316 batches')

plt.plot(sort_tr_acc)
plt.plot(sort_val_acc)
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over all batches')

stats(sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc)

num_encoders = 3
num_decoders = 2
batch_size = 64
d_model = 256
h_dim = 1024
heads = 8
gradient_clip = 0.1



model = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

sort_tr_loss = []
sort_val_loss = []
sort_tr_acc = []
sort_val_acc = []

checkpoint = torch.load('Org-sort break6.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']

train(model, loss, train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,
      sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc,
      5, 10, 157,
      save="sortOriPara1-",save_mod=10000,names=[])

loc = "Org-sort"+" "+"break0"+".pt"
torch.save({
  'epoch': 0,
  'sort_tr_loss': sort_tr_loss,
  'sort_val_loss':sort_val_loss,
  'sort_tr_acc':sort_tr_acc,
  'sort_val_acc':sort_val_acc,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict()
  }, loc)

len(sort_val_loss)

#2nd model compare-sort graphs

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/saves/orgSort'

checkpointo = torch.load('Org-sort break1.pt',map_location=device)
sort_tr_losso = checkpointo['sort_tr_loss']
sort_val_losso = checkpointo['sort_val_loss']
sort_tr_acco = checkpointo['sort_tr_acc']
sort_val_acco = checkpointo['sort_val_acc']

len(sort_val_losso)

checkpoint = torch.load('Org-sort break6.pt',map_location=device)
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']

len(sort_val_loss)

len(sort_val_loss)-len(sort_val_losso)

import matplotlib.pyplot as plt
plt.plot(sort_tr_loss)
plt.plot(sort_val_loss)
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over all batches')

import matplotlib.pyplot as plt
plt.plot(sort_tr_losso)
plt.plot(sort_val_losso)
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over first 205 batches')

plt.plot(sort_tr_acco)
plt.plot(sort_val_acco)
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over first 205 batches')

stats(sort_tr_losso, sort_val_losso, sort_tr_acco, sort_val_acco)

plt.plot(sort_val_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Validation Perplexity'])
plt.title('Perplexity over next 188 batches')

plt.plot(sort_tr_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity'])
plt.title('Perplexity over next 188 batches')

import matplotlib.pyplot as plt
plt.plot(sort_tr_loss[len(sort_tr_losso):])
plt.plot(sort_val_loss[len(sort_tr_losso):])
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over next 188 batches')

plt.plot(sort_tr_acc)
plt.plot(sort_val_acc)
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over all batches')

stats(sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc)

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/algebra'

# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...


TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"


# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

#src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"


train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

#src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
#tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

src_file_path = f"{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False)





from torch.utils.data import DataLoader

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

src_vocab.id_to_string

tgt_vocab.id_to_string

# Example batch
batch = next(iter(train_data_loader))

source = batch[0]  # source sequence
print(source.shape)
target = batch[1]  # target sequence
print(target.shape)

# example source/target pair
example_source_sequence = []

for i in source[0]:
    example_source_sequence.append(src_vocab.id_to_string[i.item()])
print(source[0])
print(example_source_sequence)

print(''.join(example_source_sequence))

example_target_sequence = []

for i in target[0]:
    example_target_sequence.append(tgt_vocab.id_to_string[i.item()])

print(example_target_sequence)

num_encoders = 1
num_decoders = 1
batch_size = 64
d_model = 256
h_dim = 512
heads = 8
gradient_clip = 0.1

model = TransformerModel(len(src_vocab), len(tgt_vocab), d_model, heads, h_dim, num_encoders, num_decoders, src_vocab.pad_id, device)
model.to(device)
loss = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

sort_tr_loss = []
sort_val_loss = []
sort_tr_acc = []
sort_val_acc = []

checkpoint = torch.load("Last break2.pt")
model.load_state_dict(checkpoint['model_state_dict'])
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

train(model, loss, train_data_loader, valid_data_loader, device, gradient_clip, src_vocab, tgt_vocab,
      sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc,
      30, 10, 157,
      save="6algebra",save_mod=10000,names=[],algebra=True)

for g in optimizer.param_groups:
    g['lr'] = 0.00001

loc = "Last"+" "+"break2"+".pt"
torch.save({
  'sort_tr_loss': sort_tr_loss,
  'sort_val_loss':sort_val_loss,
  'sort_tr_acc':sort_tr_acc,
  'sort_val_acc':sort_val_acc,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict()
  }, loc)

#plots

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/BI/saves/alg'

checkpoint = torch.load('Last break5 (1).pt',map_location=device)
sort_tr_loss = checkpoint['sort_tr_loss']
sort_val_loss = checkpoint['sort_val_loss']
sort_tr_acc = checkpoint['sort_tr_acc']
sort_val_acc = checkpoint['sort_val_acc']

len(sort_val_loss)

import matplotlib.pyplot as plt
plt.plot(sort_tr_loss)
plt.plot(sort_val_loss)
plt.xlabel('Batches')
plt.ylabel('Perplexity')
plt.legend(['Training Perplexity','Validation Perplexity'])
plt.title('Perplexity over all batches')

plt.plot(sort_tr_acc)
plt.plot(sort_val_acc)
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Validation accuracy'])
plt.title('Accuracy over all batches')

stats(sort_tr_loss, sort_val_loss, sort_tr_acc, sort_val_acc)

