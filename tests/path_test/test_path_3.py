""" Simple multi-layer perception neural network using Minpy """
import os
import argparse

import minpy.numpy as np
from minpy.nn.io import NDArrayIter
# Can also use MXNet IO here
# from mxnet.io import NDArrayIter
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from examples.utils.data_utils import get_CIFAR10_data

# Please uncomment following if you have GPU-enabled MXNet installed.
#from minpy.context import set_context, gpu
#set_context(gpu(0)) # set the global context as gpu(0)


def test_path_3():
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name, '../../examples/dataset/cifar10/cifar-10-batches-py')
    data = get_CIFAR10_data(file_path)

if __name__ == '__main__':
    test_path_3()
