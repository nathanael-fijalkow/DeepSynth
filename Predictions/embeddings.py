import logging 
import torch
import torch.nn as nn
import numpy as np

"""
IO = [[I1, ...,Ik], O]
I1, ..., Ik, O are lists
IOs = [IO1, IO2, ..., IOn]
task = (IOs1, program1)
tasks = [task1, task2, ..., taskp]
"""
class SimpleEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 ):
        super(SimpleEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension

        embedding = nn.Embedding(self.lexicon_size, output_dimension)
        self.embedding = embedding

    def forward_IOs(self, IOs):
        """
        returns a tensor of shape 
        (len(IOs), self.IOEncoder.output_dimension, self.output_dimension)
        """
        # e = self.IOEncoder.encode_IOs(IOs)
        # logging.debug("encoding size: {}".format(e.size()))
        e = self.embedding(IOs)
        logging.debug("embedding size: {}".format(e.size()))
        assert(e.size() == (len(IOs), self.IOEncoder.output_dimension, self.output_dimension))
        return e

    # def forward(self, batch_IOs):
    #     """
    #     returns a tensor of shape 
    #     (len(batch_IOs), self.IOEncoder.output_dimension, self.output_dimension)
    #     """
    #     res = torch.stack([self.forward_IOs(IOs) for IOs in batch_IOs])
    #     # assert(res.size() == (len(batch_IOs), self.IOEncoder.output_dimension, self.output_dimension))
    #     return res
