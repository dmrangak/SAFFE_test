#last update march 6 8pm
## Standard libraries
import os
import numpy as np
import random
import math
import json
from functools import partial
import config

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
#%matplotlib inline
#import matplotlib.pyplot
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
#matplotlib_inline.backend_inline.set_matplotlib_formats()
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg','pdf')



from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.types import _size

## Torchvision
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms


## Audio
from transformers import AutoProcessor, ASTModel
from datasets import load_dataset
from transformers import AutoProcessor, ASTModel

####image
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel , CLIPVisionModelWithProjection


from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from pathlib import Path
import copy

# Huggingface datasets and tokenizers

import torchmetrics
from torch.utils.tensorboard import SummaryWriter




def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask



#from overrides import has_torch_function, handle_torch_function
Tensor = torch.Tensor
def linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    tens_ops = (input, weight)
    if not torch.jit.is_scripting():
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


def scaled_dot_product(q, k, v, mask=None):
    # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64, mask 200 x 200
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 x 8 x 200 x 200
   # print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
   #     print(f"-- ADDING MASK of shape {mask.size()} --")
        scaled += mask # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1) # 30 x 8 x 200 x 200
   # attention= attention.permute(0, 1, 3, 2)
  #  print("attention",attention.shape)
    values = torch.matmul(attention, v) # 30 x 8 x 200 x 64
 #   print("values",values.shape)
    return values, attention

#def calqulate_bias_weight_image(q, k, v, mask=None):

class PositionwiseFeedForward(nn.Module):
     def __init__(self, d_model, hidden, drop_prob=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.linear1 = nn.Linear(d_model, hidden)
         self.linear2 = nn.Linear(hidden, d_model)
         self.relu = nn.ReLU()
#torch.nn.functional.relu (x, inplace = True)
         self.dropout = nn.Dropout(p=drop_prob)
         #print("ffffffffffffffffffffffffffffffffffffffff",hidden)

     def forward(self, x):
         #  x: 30 x 200 x 512
         x = self.linear1(x) #30 x 200 x 2048
         #print(f"x after first linear layer: {x.size()}")
         
         x = self.relu(x) #30 x 200 x 2048
         #print(f"x after relu layer: {x.size()}")
         x = self.dropout(x) #30 x 200 x 2048
         #print(f"x after dropout layer: {x.size()}")
         x = x.clone().detach()
         x = self.linear2(x) #30 x 200 x 512
        # print(f"x after 2nd linear layer: {x.size()}")
         return x #30 x 200 x 512




class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # 512
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # 512

    def forward(self, inputs):
        # inputs : 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1]
        #print(f"dims: {dims}")
        mean = inputs.mean(dim=dims, keepdim=True) #30 x 200 x 1
        #print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # 30 x 200 x 512
        std = (var + self.eps).sqrt() # 30 x 200 x 512
        #print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std # 30 x 200 x 512
        #print(f"y: {y.size()}")
        out = self.gamma * y  + self.beta  # 30 x 200 x 512
       # print(f"out: {out.size()}")
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads,count):
        super().__init__()
        self.count=count
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # 1536
        self.linear_layer = nn.Linear(d_model, d_model)
        self.embed_dim=d_model
    def forward(self, x, mask,image_query_bias,image_key_bias,image_value_bias,image_query_weight,image_key_weight,image_value_weight,image_outputs):
        mask=None
        self.image_outputs=image_outputs
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        #print(f"y.size(): {x.size()}")

        

        # ####################
        if self.count == 0:
        # ######################################
        # This is inline in_proj function with in_proj_weight and in_proj_bias
           _b = image_query_bias
           
           _start = 0
           _end = self.embed_dim
           _w = image_query_weight[_start:_end, :]
           if _b is not None:
                _b = _b[_start:_end]
           self.q = linear(self.image_outputs, _w, _b)
           #print("final q value shape", self.q.shape)
           #print("final q value ", self.q)
           self.q=self.q.reshape(batch_size, sequence_length, self.num_heads,self.head_dim)
           self.q=self.q.permute(0, 2, 1, 3)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
           _b = image_key_bias
           _start = 0
           _end = self.embed_dim
           _w = image_key_weight[_start:_end, :]
           if _b is not None:
                 _b = _b[_start:_end]
           self.k = linear(self.image_outputs, _w, _b)
           self.k=self.k.reshape(batch_size, sequence_length, self.num_heads,self.head_dim)
           self.k=self.k.permute(0, 2, 1, 3)
            #print("final k value shape", self.k.shape)
           # print("final k value ", self.k)

         # This is inline in_proj function with in_proj_weight and in_proj_bias
           _b = image_value_bias
           _start = 0
           _end = self.embed_dim
           _w = image_value_weight[_start:_end, :]
           if _b is not None:
             _b = _b[_start:_end]
           self.v = linear(self.image_outputs, _w, _b)
           self.v=self.v.reshape(batch_size, sequence_length, self.num_heads,self.head_dim)
           self.v=self.v.permute(0, 2, 1, 3)
           #print("final v value shape", self.v.shape)
           #print("final v value ", self.v)

        # ###################################
        else:
           qkv = self.qkv_layer(x) # 30 x 200 x 1536
           #qkv = qkv.clone().detach()
           #print(f"qkv.size(): {qkv.size()}")
           qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
           #print(f"qkv after reshape .size(): {qkv.size()}")
           qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
           #print(f"qkv after permutation: {qkv.size()}")
           self.q, self.k, self.v = qkv.chunk(3, dim=-1)



        #print(f"q: {self.q.size()}, k:{self.k.size()}, v:{self.v.size()}")
        values, attention = scaled_dot_product(self.q, self.k, self.v, mask) # values: 30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        #values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        values= values.permute(0,2,1,3).reshape(batch_size, sequence_length, self.num_heads* self.head_dim)
        #print(f"values after reshaping: {values.size()}")
        out = self.linear_layer(values) # 30 x 200 x 512
        out = out.detach().clone()
        #print(f"out after passing through linear layer: {out.size()}")
        return out # 30 x 200 x 512


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads,count):
        super().__init__()
        self.count=count
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.embed_dim=d_model

        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask, image_query_bias,image_key_bias,image_value_bias,image_query_weight,image_key_weight,image_value_weight, 
                audio_key_bias,audio_value_bias,audio_key_weight,audio_value_weight,constant_y):
        mask=None
       # self.image_model=image_model
        #print("maithri",x.shape)
        batch_size, sequence_length_x, d_model = x.shape # 30 x 200 x 512
        #print(f"x.size(): {x.size()}")
        #print(f"y.size(): {y.size()}")
        #print("maithri",sequence_length_x)
##############################################
        batch_size, sequence_length_y, d_model = y.shape
      #  if sequence_length_x >= sequence_length_y:
       #    sequence_length = sequence_length_x
       # else:
        sequence_length=sequence_length_y
 #       batch_size=config['batch_size']
 #       sequence_length=config['sequence_length_audio']
##################################################
          ####################
        _b = image_query_bias
        _start = 0
        _end = self.embed_dim
        _w = image_query_weight[_start:_end, :]
        if _b is not None:
         _b = _b[_start:_end]
        self.q = linear(constant_y, _w, _b)
            #print("final q value shape", self.q.shape)
            #print("final q value ", self.q)
        self.q=self.q.reshape(batch_size, sequence_length_y, self.num_heads,self.head_dim)
        self.q=self.q.permute(0, 2, 1, 3)

        if self.count == 0:
        # ######################################
         # This is inline in_proj function with in_proj_weight and in_proj_bias

  
         # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = audio_key_bias
           
            _start = 0
            _end = self.embed_dim
            _w = audio_key_weight[_start:_end, :]
            if _b is not None:
             _b = _b[_start:_end]
            self.k = linear(x, _w, _b)
            self.k=self.k.reshape(batch_size, sequence_length_x, self.num_heads,self.head_dim)
            self.k=self.k.permute(0, 2, 1, 3)
            #print("final k value shape", self.k.shape)
           # print("final k value ", self.k)

        # # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = audio_value_bias
            _start = 0
            _end = self.embed_dim
            _w = audio_value_weight[_start:_end, :]
            if _b is not None:
              _b = _b[_start:_end]
            self.v = linear(x, _w, _b)
            self.v=self.v.reshape(batch_size, sequence_length_x, self.num_heads,self.head_dim)
            self.v=self.v.permute(0, 2, 1, 3)
            #print("final v value shape", self.v.shape)
            #print("final v value ", self.v)

        # ###################################
        else:
           kv = self.kv_layer(x) # 30 x 200 x 1024

           #print(f"kv.size(): {kv.size()}")
           #self.q = self.q_layer(y) # 30 x 200 x 512
           #print(f"q.size(): {self.q.size()}")
           kv = kv.reshape(batch_size, sequence_length_x, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
           #self.q = self.q.reshape(batch_size, sequence_length_y, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
           kv = kv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 128
           #self.q = self.q.permute(0, 2, 1, 3) # 30 x 8 x 200 x 64
           self.k, self.v = kv.chunk(2, dim=-1) # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64

        values, attention = scaled_dot_product(self.q, self.k, self.v, mask) #  30 x 8 x 200 x 64
        #print(f"values: {values.size()}, attention:{attention.size()}")
        #values = values.reshape(batch_size, sequence_length, d_model) #  30 x 200 x 512
        values= values.permute(0,2,1,3).reshape(batch_size, sequence_length, self.num_heads* self.head_dim)
        out = self.linear_layer(values) 
        out = out.detach().clone() #  30 x 200 x 512
        #print(f"out after passing through linear layer: {out.size()}")
        return out  #  30 x 200 x 512


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,count):
        super(DecoderLayer, self).__init__()
        self.count=count

        #print("Layer number",count)
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads,count=self.count)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads,count=self.count)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight,image_query_bias, image_key_bias,image_value_bias,
                image_query_weight,image_key_weight,image_value_weight,image_outputs,constant_y):
        #self.model_audio=model_audio
      
        self.image_outputs=image_outputs
        self.mask=None
        #print("Layer number",self.count)
        _y = y # 30 x 200 x 512
        #print("MASKED SELF ATTENTION")

        y = self.self_attention(y,self.mask, image_query_bias,image_key_bias,image_value_bias,image_query_weight,image_key_weight,image_value_weight,image_outputs=self.image_outputs) # 30 x 200 x 512
        #print("DROP OUT 1")
        y = self.dropout1(y) # 30 x 200 x 512
        #print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y) # 30 x 200 x 512

        _y = y # 30 x 200 x 512
        #print("CROSS ATTENTION")
        #if self.count == 0:
        y = self.encoder_decoder_attention(x, y, self.mask,image_query_bias,image_key_bias,image_value_bias,image_query_weight,image_key_weight,image_value_weight,
                                        audio_key_bias,audio_value_bias,audio_key_weight,audio_value_weight,constant_y) #30 x 200 x 512
           #print("DROP OUT 2")  #30 x 200 x 512
        y = self.dropout2(y)
           #print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)  #30 x 200 x 512

        _y = y  #30 x 200 x 512
        #print("FEED FORWARD 1")
        y = self.ffn(y) #30 x 200 x 512
        #print("DROP OUT 3")
        y = self.dropout3(y) #30 x 200 x 512
        #print("Maithri ranga kulasekara")
        y = self.norm3(y + _y) #30 x 200 x 512
        return y #30 x 200 x 512

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight,image_query_bias, image_key_bias,image_value_bias,image_query_weight,image_key_weight,image_value_weight,image_outputs,constant_y= inputs
        for module in self._modules.values():
            y = module(x, y, mask,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight,image_query_bias, image_key_bias,image_value_bias,
                image_query_weight,image_key_weight,image_value_weight,image_outputs,constant_y) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob, _)
                                          for _ in range(num_layers)])
        #self.layers = DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob,2 )
    def forward(self, x, y,mask,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight,image_query_bias, image_key_bias,image_value_bias,
                image_query_weight,image_key_weight,image_value_weight,image_outputs,constant_y):
        #x : 30 x 200 x 512
        #y : 30 x 200 x 512
        #mask : 200 x 200
        y = self.layers(x, y, mask,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight, image_query_bias, image_key_bias,image_value_bias,
                image_query_weight,image_key_weight,image_value_weight,image_outputs,constant_y)
        return y #30 x 200 x 512
    

class Transformer(nn.Module):
    def __init__(self,
                d_model,
                ffn_hidden,
                num_heads,
                drop_prob,
                num_layers,
                kn_vocab_size):
        super().__init__()
        self.new_kn_vocab_size=2*50
   #    self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder_image = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
   #     self.decoder_image = Decoder_image(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
       # kn_vocab_size=2*(outputs_audio.hidden_states[9].size(dim=1) + image_outputs.hidden_states[9].size(dim=1) )
        self.d_model=d_model
        num_neurons =self.d_model*100
        self.linear = nn.Linear(num_neurons,self.d_model)



    def forward(self,batch_size,audio_query_bias,audio_key_bias,audio_value_bias,audio_query_weight,audio_key_weight,audio_value_weight,image_query_bias, image_key_bias,image_value_bias,
                image_query_weight,image_key_weight,image_value_weight,outputs_audio, image_outputs,image_pooled_output): # x, y are batch of sentences
 #       x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
 #       out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        
       # self.kn_vocab_size=2*(outputs_audio.hidden_states[9].size(dim=1) )
       # self.kn_vocab_size2=outputs_audio.hidden_states[9].size(dim=1) 
       # self.linear = nn.Linear(self.kn_vocab_size,1) # output is one value 
       # self.linear = nn.Linear(self.kn_vocab_size,self.kn_vocab_size2)
        
 #       self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##################################
        ##########image to Audio
 
         #mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
         #mask = torch.triu(mask, diagonal=1)
#        mask= None
#        decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
#        out = decoder(x, y, mask)

        self.x_a_10=outputs_audio[0]
        self.y_a_10=image_outputs[0]
        self.y_a_10_2=self.y_a_10
        

        self.x_a_9=outputs_audio[1]
        self.y_a_9=image_outputs[1]
        self.y_a_9_2=self.y_a_9
 ############################ image to Audio
        mask= None

        decoder_image_10= self.decoder_image( self.x_a_10, self.y_a_10, mask,audio_query_bias[0],audio_key_bias[0],audio_value_bias[0],audio_query_weight[0],audio_key_weight[0],audio_value_weight[0],
                                              image_query_bias[0],image_key_bias[0],image_value_bias[0],image_query_weight[0],image_key_weight[0],image_value_weight[0],image_outputs[0],self.y_a_10_2)
        decoder_image_9= self.decoder_image( self.x_a_9, self.y_a_9, mask,audio_query_bias[1],audio_key_bias[1],audio_value_bias[1],audio_query_weight[1],audio_key_weight[1],audio_value_weight[1],
                                             image_query_bias[1],image_key_bias[1],image_value_bias[1],image_query_weight[1],image_key_weight[1],image_value_weight[1],image_outputs[1],self.y_a_9_2)
        #print("##########image to Audio#########################################")
############################  Audio to image
       # decoder_image_9 = self. decoder_image(self.x_i_9, self.y_i_9, mask)
       # decoder_image_10 = self. decoder_image(self.x_i_10, self.y_i_10, mask)
        #decoder_image_10=image_outputs[0]
       # decoder_image_9=image_outputs[1]

################################################
       # print("image to Audio Bottleneck _1 ",decoder_audio_9.shape)
       # print("image to Audio Bottleneck _2 ",decoder_audio_10.shape)
       # print("Audio to image  Bottleneck _1 ",decoder_image_9.shape)
        #print("Audio to image  Bottleneck _2 ",decoder_image_10.shape)
       # avg_decoder= torch.add(decoder_image_10, decoder_image_11)
        #avg_decoder = torch.mul(avg_decoder, 0.5)  # 50,768 

        #concat_all= torch.cat((decoder_audio_9,decoder_audio_10,decoder_image_9,decoder_image_10),1)
        concat_all= torch.cat((decoder_image_10,decoder_image_9),1)  # 1,50,768 
        #concat_all= decoder_image_10
        #print("Audio to image  Bottleneck _2 ",concat_all.shape)
        #concat_all= concat_all.permute(0,2,1) # 1,768,50
        #batchsize = config['batch_size'] - 1
        self.batchsize= batch_size
        #self.batchsize= 10
        y = concat_all.reshape(self.batchsize, 76800)
        #print("Audio to image  Bottleneck _3 ", y.shape)
        #print("##########image to Audio#########################################",concat_all.shape)
        #pool = nn.AdaptiveAvgPool2d(output_size=(1, 768))
        #y = pool(concat_all)
        #print("##########maithri ranga#########################################",y.shape)
        out = F.relu(self.linear(y)) # 1, 768,1 
        #out=out.permute(0,2,1)     #   1,1,768 
        #print("Audio to image  Bottleneck  ",out.shape)
         # Initialize the parameters  
        for p in self.decoder_image.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  
        
        
        return out
    
