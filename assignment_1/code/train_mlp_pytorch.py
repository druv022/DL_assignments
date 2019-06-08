"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '400,600'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 6000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

LR_DROP = [2000]

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  pred_index = np.argmax(predictions,axis=1)
  target_index = np.argmax(targets, axis=1)
  correct = np.count_nonzero(np.equal(pred_index,target_index),axis=0)
  accuracy = correct/targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cpu")

  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  train = cifar10['train']

  # TODO: verify hardcoded
  input_features = 3*32*32
  output_class = 10

  model = MLP(input_features, dnn_hidden_units, output_class)
  model.to(device)

  optimizer = torch.optim.SGD(model.parameters(),lr=FLAGS.learning_rate, momentum=0.9)

  cross_e = torch.nn.CrossEntropyLoss()
  loss_iter = []
  loss_mean = []
  acc_iter = []

  loss_sum = 0.0

  for iter_n in np.arange(0,FLAGS.max_steps):
    x, t = train.next_batch(FLAGS.batch_size)
    x = torch.tensor(x).to(device)
    x = x.view(FLAGS.batch_size,-1)
    t = np.argwhere(t>0)
    t = torch.from_numpy(t[:,1]).to(device)

    if (iter_n+1) in LR_DROP:
      for item in optimizer.param_groups:
        item['lr'] = FLAGS.learning_rate/10

    optimizer.zero_grad()

    model.train()
    y = model.forward(x)

    loss = cross_e(y,t.long())
    loss_iter.append(loss.item())
    loss_sum += loss.item()
    loss_mean.append(loss_sum/(iter_n+1))

    loss.backward()
    optimizer.step()
    # print("Iter: ", iter_n, " | Loss: ", loss.item())

    if (iter_n+1) % int(FLAGS.eval_freq) == 0:
      test = cifar10['test']

      acc = []
      with torch.no_grad():
        for _ in np.arange(0,(test.num_examples//FLAGS.batch_size)):
          x, t = test.next_batch(FLAGS.batch_size)
          x = torch.tensor(x).to(device)
          x = x.view(FLAGS.batch_size,-1)
          t = torch.tensor(t).to(device)

          model.eval()
          y = model.forward(x)
          acc.append(accuracy(y,t))
        acc_iter.append(np.mean(acc))
        print(np.mean(acc))

  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.plot(loss_iter,'b-',alpha=0.3, label='per step')
  plt.plot(loss_mean,'r-', label='moving average')
  plt.legend()
  plt.show()

  plt.xlabel("Iterations(x100)")
  plt.ylabel("Accuracy")
  plt.plot(acc_iter,'g-')
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()