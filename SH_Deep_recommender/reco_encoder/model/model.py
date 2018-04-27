# Copyright (c) 2017 NVIDIA Corporation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

def activation(input, kind):
  #print("Activation: {}".format(kind))
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return F.sigmoid(input)
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'swish':
    return input*F.sigmoid(input)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')

def MSEloss(inputs, targets, size_avarage=False):
  mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(size_average=size_avarage)
  return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_avarage else num_ratings

class AutoEncoder(nn.Module):
  def __init__(self, layer_sizes, nl_type='selu', is_constrained=False, dp_drop_prob=0.0, last_layer_activations=True):
    """
    Describes an AutoEncoder model
    :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
    For example: [10000, 1024, 512] will result in:
      - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
      - decoder 2 layers: 512x1024 and 1024x10000.
    :param nl_type: (default 'selu') Type of no-linearity
    :param is_constrained: (default: True) Should constrain decoder weights
    :param dp_drop_prob: (default: 0.0) Dropout drop probability
    :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
    """
    super(AutoEncoder, self).__init__()
    self._dp_drop_prob = dp_drop_prob
    self._last_layer_activations = last_layer_activations
    if dp_drop_prob > 0:
      self.drop = nn.Dropout(dp_drop_prob)
    self._last = len(layer_sizes) - 2
    self._nl_type = nl_type

    self.encode_w = nn.ParameterList(
        # default: 7142,512,512,1024
        [nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])

    for ind, w in enumerate(self.encode_w):
        weight_init.xavier_uniform(w)

    self.encode_b = nn.ParameterList(
        [nn.Parameter(torch.zeros(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)])

    reversed_enc_layers = list(reversed(layer_sizes))
    # default: 1024,512,512,7142

    self.is_constrained = is_constrained
    if not is_constrained:
        self.decode_w = nn.ParameterList()
        for i in range(len(reversed_enc_layers) - 1):
            if i != 0:
                self.decode_w.append(nn.Parameter(torch.rand(reversed_enc_layers[i + 1], reversed_enc_layers[i] * 2)))
            else:
                self.decode_w.append(nn.Parameter(torch.rand(reversed_enc_layers[i + 1], reversed_enc_layers[i])))

        for ind, w in enumerate(self.decode_w):
            weight_init.xavier_uniform(w)
    self.decode_b = nn.ParameterList(
        [nn.Parameter(torch.zeros(reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

    print("******************************")
    print("******************************")
    print(layer_sizes)
    print("Dropout drop probability: {}".format(self._dp_drop_prob))
    print("Encoder pass:")
    for ind, w in enumerate(self.encode_w):
        print(w.data.size())
        print(self.encode_b[ind].size())
    print("Decoder pass:")
    if self.is_constrained:
        print('Decoder is constrained')
        for ind, w in enumerate(list(reversed(self.encode_w))):
            print(w.transpose(0, 1).size())
            print(self.decode_b[ind].size())
    else:
        for ind, w in enumerate(self.decode_w):
            print(w.data.size())
            print(self.decode_b[ind].size())
    print("******************************")
    print("******************************")


  def encode(self, x):
    x1 = activation(input=F.linear(input=x, weight=self.encode_w[0],
                                   bias=self.encode_b[0]), kind=self._nl_type)
    x2 = activation(input=F.linear(input=x1, weight=self.encode_w[1],
                                   bias=self.encode_b[1]), kind=self._nl_type)
    z = activation(input=F.linear(input=x2, weight=self.encode_w[2],
                                   bias=self.encode_b[2]), kind=self._nl_type)

    if self._dp_drop_prob > 0: # apply dropout only on code layer
      z = self.drop(z)
    # print("z",z.shape)
    return x1, x2, z

  def decode(self, x):
    encode_1, encode_2, z = self.encode(x)
    # print(encode_1.shape, encode_2.shape, z.shape)

    decode_3 = activation(input=F.linear(input=z, weight=self.decode_w[0],
                                         bias=self.decode_b[0]), kind="selu")
    decode_2_input = torch.cat([decode_3, encode_2], dim=1)
    # print("de2 input", decode_2_input.shape)
    decode_2 = activation(input=F.linear(input=decode_2_input, weight=self.decode_w[1]*2,
                                         bias=self.decode_b[1]*2), kind="selu")
    decode_1_input = torch.cat([decode_2, encode_1], dim=1)
    # print("de1 input", decode_1_input.shape)

    decode_1 = activation(input=F.linear(input=decode_1_input, weight=self.decode_w[2]*2,
                                         bias=self.decode_b[2]*2), kind="none")

    # print(decode_3.shape, decode_2.shape, decode_1.shape)

    return decode_1

  def forward(self, x):
    # print("x",x.shape)
    return self.decode(x)

