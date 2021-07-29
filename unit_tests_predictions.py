import logging
import unittest
import random
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from program import Program, Function, Variable, BasicPrimitive, New
from dsl import DSL
from pcfg_logprob import LogProbPCFG
from Predictions.IOencodings import FixedSizeEncoding, VariableSizeEncoding
from Predictions.embeddings import SimpleEmbedding#, RecurrentEmbedding

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

class TestSum(unittest.TestCase):
    def test_encoding(self):
        size_max = 2 # maximum number of elements in an input (= list)
        nb_inputs_max = 5 # maximum number of inputs 
        lexicon = list(range(30))

        IOEncoder = FixedSizeEncoding(
            nb_inputs_max = nb_inputs_max,
            lexicon = lexicon,
            size_max = size_max,
            )

        encoding_output_dimension = 15 # fixing the dimension, 
        # only useful for VariableSizeEncoding

        IOEncoder2 = VariableSizeEncoding(
            nb_inputs_max = nb_inputs_max,
            lexicon = lexicon,
            output_dimension = encoding_output_dimension,
            )

        IO1 = [[[11,20], [3]], [12,2]] 
        IO2 = [[[12,23], [2,15], [4,2], [0]], [2]] 

        res = IOEncoder.encode_IO(IO1)
        self.assertTrue(len(res) == IOEncoder.output_dimension)
        res = IOEncoder2.encode_IO(IO1)
        self.assertTrue(len(res) == IOEncoder2.output_dimension)
        res = IOEncoder.encode_IO(IO2)
        self.assertTrue(len(res) == IOEncoder.output_dimension)
        res = IOEncoder2.encode_IO(IO2)
        self.assertTrue(len(res) == IOEncoder2.output_dimension)

        IOs = [IO1, IO2]

        res = IOEncoder.encode_IOs(IOs)
        self.assertTrue(res.size() == (len(IOs), IOEncoder.output_dimension))
        res = IOEncoder2.encode_IOs(IOs)
        self.assertTrue(res.size() == (len(IOs), IOEncoder2.output_dimension))

    def test_embedding(self):
        size_max = 2 # maximum number of elements in an input (= list)
        nb_inputs_max = 5 # maximum number of inputs 
        lexicon = list(range(30))

        IOEncoder = FixedSizeEncoding(
            nb_inputs_max = nb_inputs_max,
            lexicon = lexicon,
            size_max = size_max,
            )

        encoding_output_dimension = 15 # fixing the dimension, 
        # only useful for VariableSizeEncoding

        IOEncoder2 = VariableSizeEncoding(
            nb_inputs_max = nb_inputs_max,
            lexicon = lexicon,
            output_dimension = encoding_output_dimension,
            )

        IO1 = [[[11,20], [3], [2], [23]], [12,2]] 
        IO2 = [[[12,23], [2,15], [4,2], [0]], [2]] 
        IOs = [IO1, IO2]
        batch_IOs = [IOs, IOs, IOs]

        embedding_output_dimension = 30

        IOEmbedder = SimpleEmbedding(
            IOEncoder = IOEncoder,
            output_dimension = embedding_output_dimension,
            )

        print("output dimension of the encoder", IOEncoder.output_dimension)
        res = IOEmbedder.forward_IOs(IOs)
        self.assertTrue(res.size() == (len(IOs), IOEncoder.output_dimension, IOEmbedder.output_dimension))

        # res = IOEmbedder.forward(batch_IOs)
        # self.assertTrue(res.size() == (len(batch_IOs), IOEncoder.output_dimension, IOEmbedder.output_dimension))

    # def test_predictions_noinputs(self):
    #     primitive_types = {
    #         "if": Arrow(BOOL, Arrow(INT, INT)),
    #         "+": Arrow(INT, Arrow(INT, INT)),
    #         "0": INT,
    #         "1": INT,
    #         "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
    #         "lt": Arrow(INT, Arrow(INT, BOOL)),
    #     }

    #     semantics = {
    #         "if": lambda b: lambda x: lambda y: x if b else y,
    #         "+": lambda x: lambda y: x + y,
    #         "0": 0,
    #         "1": 1,
    #         "and": lambda b1: lambda b2: b1 and b2,
    #         "lt": lambda x: lambda y: x <= y,
    #     }

    #     template_dsl = DSL(semantics, primitive_types)
    #     type_request = INT
    #     template_cfg = template_dsl.DSL_to_CFG(type_request=type_request, 
    #                                   upper_bound_type_size=4,
    #                                   max_program_depth=4, 
    #                                   min_variable_depth=2,
    #                                   n_gram=2)

    #     list_variables = [
    #     Variable(i, type_, probability={})
    #     for i,type_ in enumerate(type_request.arguments())
    #     ]

    #     H = 128 # hidden size of neural network
    #     lexicon = list(range(10)) # all elements in range(10)
    #     fe = RecurrentFeatureExtractor(lexicon=lexicon,
    #                                    H=H,
    #                                    bidirectional=True)

    #     PCFG_predictor = PCFG_Predictor(
    #         fe,
    #         template_cfg=template_cfg
    #         )

    #     Q_predictor = Q_Predictor(
    #         fe,
    #         template_dsl=template_dsl,
    #         template_cfg=template_cfg,
    #         list_variables=list_variables,
    #         )

    #     programs = [
    #         Function(BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))),[BasicPrimitive("0", INT),BasicPrimitive("1", INT)], INT),
    #         ]

    #     x = [] # input
    #     y = [] # output
    #     ex = (x,y) # a single input/output example

    #     tasks = [[ex]]

    #     PCFG_predictor.train(programs, tasks)
    #     PCFG_predictor.test(programs, tasks)
    #     Q_predictor.train(programs, tasks)
    #     Q_predictor.test(programs, tasks)

    # def test_predictions_with_inputs(self):
    #     t0 = PolymorphicType('t0')
    #     t1 = PolymorphicType('t1')
    #     primitive_types = {
    #         "if": Arrow(BOOL, Arrow(INT, INT)),
    #         "+": Arrow(INT, Arrow(INT, INT)),
    #         "0": INT,
    #         "1": INT,
    #         "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
    #         "lt": Arrow(INT, Arrow(INT, BOOL)),
    #         "map": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
    #     }

    #     semantics = {
    #         "if": lambda b: lambda x: lambda y: x if b else y,
    #         "+": lambda x: lambda y: x + y,
    #         "0": 0,
    #         "1": 1,
    #         "and": lambda b1: lambda b2: b1 and b2,
    #         "lt": lambda x: lambda y: x <= y,
    #         "map": lambda l: list(map(f, l)),
    #     }

    #     template_dsl = DSL(semantics, primitive_types)
    #     type_request = Arrow(List(INT), List(INT))
    #     template_cfg = template_dsl.DSL_to_CFG(type_request=type_request, 
    #                                   upper_bound_type_size=10,
    #                                   max_program_depth=4, 
    #                                   min_variable_depth=1,
    #                                   n_gram = 1)

    #     H = 128 # hidden size of neural network
    #     lexicon = list(range(10))
    #     fe = RecurrentFeatureExtractor(lexicon=lexicon,
    #                                    H=H,
    #                                    bidirectional=True)

    #     list_variables = [
    #     Variable(i, type_, probability={})
    #     for i,type_ in enumerate(type_request.arguments())
    #     ]

    #     PCFG_predictor = PCFG_Predictor(
    #         fe,
    #         template_cfg=template_cfg
    #         )

    #     Q_predictor = Q_Predictor(
    #         fe,
    #         template_dsl=template_dsl,
    #         template_cfg=template_cfg,
    #         list_variables=list_variables,
    #         )

    #     programs = [
    #         Function(
    #             BasicPrimitive("map", Arrow(Arrow(INT, INT), Arrow(List(INT), List(INT)))),
    #             [
    #             Function(
    #                 BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #                 [BasicPrimitive("1", INT)], 
    #                 Arrow(INT, INT)
    #                 ),
    #             Variable(0, List(INT))
    #             ],
    #             List(INT)
    #             ),

    #         Function(
    #             BasicPrimitive("map", Arrow(Arrow(INT, INT), Arrow(List(INT), List(INT)))),
    #             [
    #             Function(
    #                 BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #                 [Function(
    #                     BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #                     [BasicPrimitive("1", INT), BasicPrimitive("1", INT)],
    #                     INT),
    #                 ],
    #                 Arrow(INT, INT)
    #                 ),
    #             Variable(0, List(INT))
    #             ],
    #             List(INT)
    #             )
    #         ]

    #     # each task is a list of I/O
    #     # each I/O is a tuple of input, output
    #     # each output is a list whose members are elements of self.lexicon
    #     # each input is a tuple of lists, and each member of each such list is an element of self.lexicon

    #     x = ([4,4,2],) # input
    #     y = [5,5,3] # output
    #     ex1 = (x,y) # a single input/output example

    #     x = ([7,1],) # input
    #     y = [8,2] # output
    #     ex2 = (x,y) # a single input/output example

    #     task1 = [ex1,ex2] # a task is a list of input/outputs

    #     x = ([4,4,2],) # input
    #     y = [6,6,4] # output
    #     ex1 = (x,y) # a single input/output example

    #     task2 = [ex1] # a task is a list of input/outputs

    #     self.assertTrue fe.forward_one_task(task1).shape == torch.Size([H])
    #     # batched forward pass - test cases
    #     self.assertTrue fe.forward([task1,task2]).shape == torch.Size([2,H])
    #     self.assertTrue torch.all( fe.forward([task1,task2])[0] == fe.forward_one_task(task1) )
    #     self.assertTrue torch.all( fe.forward([task1,task2])[1] == fe.forward_one_task(task2) )

    #     # pooling of examples happens through averages - check via this self.assertTrue
    #     self.assertTrue(torch.stack([fe.forward_one_task(task1),fe.forward_one_task(task1)],0).mean(0) - fe.forward_one_task(task1)).abs().max() < 1e-5
    #     self.assertTrue(torch.stack([fe.forward_one_task(task1),fe.forward_one_task(task2)],0).mean(0) - fe.forward_one_task(task1)).abs().max() > 1e-5

    #     tasks = [task1,task2]

    #     PCFG_predictor.train(programs, tasks)
    #     PCFG_predictor.test(programs, tasks)
    #     Q_predictor.train(programs, tasks)
    #     Q_predictor.test(programs, tasks)

if __name__ == "__main__":
    unittest.main(verbosity=2)
