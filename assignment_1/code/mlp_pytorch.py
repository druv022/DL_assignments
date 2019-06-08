"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    self.layer = {}
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    num_input = n_inputs
    if n_hidden is None:
      self.layer['output'] = nn.Linear(n_inputs, n_classes)
      self.layer['output'].weight.data.normal_(0.0,1e-2)
      self.layer['output'].bias.data.fill_(0)
    else:
      
      for i,num in enumerate(n_hidden):
        self.layer['hidden_'+str(i+1)] = nn.Linear(num_input,num)
        self.layer['hidden_'+str(i+1)].weight.data.normal_(0.0,1e-2)
        self.layer['hidden_'+str(i+1)].bias.data.fill_(0)
        # self.layer['BatchNorm_'+str(i+1)] = nn.BatchNorm1d(num)
        
        self.layer['ReLU_'+str(i+1)] = nn.ReLU()

        num_input = num
          
      self.layer['output'] = nn.Linear(num_input, n_classes)
      self.layer['output'].weight.data.normal_(0.0,1e-2)
      self.layer['output'].bias.data.fill_(0)
    
    self.modules = nn.ModuleDict(self.layer)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = None
    if self.n_hidden is None:
      out = self.layer['output'](x)
    else:
      num_layers = len(self.n_hidden)
      for i in range(num_layers):
        if i == 0:
          out = self.layer['hidden_'+str(i+1)](x)
        else:
          out = self.layer['hidden_'+str(i+1)](out)
        
        # out = self.layer['BatchNorm_'+str(i+1)](out)
        out = self.layer['ReLU_'+str(i+1)](out)

      out = self.layer['output'](out)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
