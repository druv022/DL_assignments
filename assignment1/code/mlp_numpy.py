"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    self.layer = {}
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    num_input = n_inputs
    if n_hidden is None:
      self.layer['output'] = LinearModule(n_inputs, n_classes)
    else:
      
      for i,num in enumerate(n_hidden):
        self.layer['hidden_'+str(i+1)] = LinearModule(num_input,num)
        self.layer['ReLU_'+str(i+1)] = ReLUModule()

        num_input = num
          
      self.layer['output'] = LinearModule(num_input, n_classes)
      self.layer['Softmax'] = SoftMaxModule()


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
      out = self.layer['output'].forward(x)
    else:
      num_layers = len(self.n_hidden)
      for i in np.arange(0,num_layers):
        if i == 0:
          out = self.layer['hidden_'+str(i+1)].forward(x)
        else:
          out = self.layer['hidden_'+str(i+1)].forward(out)
        
        out = self.layer['ReLU_'+str(i+1)].forward(out)

      out = self.layer['output'].forward(out)
      out = self.layer['Softmax'].forward(out)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.layer['Softmax'].backward(dout)
    out = self.layer['output'].backward(out)
    if self.n_hidden is not None:
      num_layers = len(self.n_hidden)
      for i in np.arange(num_layers,0,-1):
        out = self.layer['ReLU_'+str(i)].backward(out)
        out = self.layer['hidden_'+str(i)].backward(out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
