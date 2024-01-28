import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

import neuromancer.slim as slim
import neuromancer.modules.rnn as rnn
from neuromancer.modules.activations import soft_exp, SoftExponential, SmoothedReLU
from neuromancer.modules.blocks import Block, Dropout


class Model(Block):
    """
    Multi-Layer Perceptron with dropout consistent with blocks interface
    """

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        nonlinparam={},
        hsizes=[64],
        linargs=dict(),
        dropout=0.0,
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList(
            [nonlin(**nonlinparam) for k in range(self.nhidden)] + [nn.Identity()]
        )
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )
        self.dropout = nn.ModuleList(
            [
                Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
                for _ in range(self.nhidden)
            ]
            + [nn.Identity()]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin, drop in zip(self.linear, self.nonlin, self.dropout):
            x = drop(nlin(lin(x)))
        return x
