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
from embedding import RecurrentFeatureExtractor
from pcfg_predictions import PCFG_Predictor
from Q_predictions import Q_Predictor

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

class TestSum(unittest.TestCase):
    def test_predictions_noinputs(self):
        primitive_types = {
            "if": Arrow(BOOL, Arrow(INT, INT)),
            "+": Arrow(INT, Arrow(INT, INT)),
            "0": INT,
            "1": INT,
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "lt": Arrow(INT, Arrow(INT, BOOL)),
        }

        semantics = {
            "if": lambda b: lambda x: lambda y: x if b else y,
            "+": lambda x: lambda y: x + y,
            "0": 0,
            "1": 1,
            "and": lambda b1: lambda b2: b1 and b2,
            "lt": lambda x: lambda y: x <= y,
        }

        template_dsl = DSL(semantics, primitive_types)
        type_request = INT
        template_cfg = template_dsl.DSL_to_CFG(type_request=type_request, 
                                      upper_bound_type_size=4,
                                      max_program_depth=4, 
                                      min_variable_depth=2,
                                      n_gram = 1)

        H = 128 # hidden size of neural network

        fe = RecurrentFeatureExtractor(lexicon=list(range(10)),
                                       H=H,
                                       bidirectional=True)

        PCFG_predictor = PCFG_Predictor(
            fe,
            template_cfg=template_cfg
            )

        Q_predictor = Q_Predictor(
            fe,
            template_dsl=template_dsl,
            template_cfg=template_cfg
            )

        programs = [
            Function(BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))),[BasicPrimitive("0", INT),BasicPrimitive("1", INT)], INT),
            ]

        x = [] # input
        y = [] # output
        ex = (x,y) # a single input/output example

        tasks = [[ex]]

        PCFG_predictor.train(programs, tasks)
        PCFG_predictor.test(programs, tasks)
        Q_predictor.train(programs, tasks)
        Q_predictor.test(programs, tasks)

    # def test_predictions_with_inputs(self):
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
    #                                   n_gram = 1)

    #     H = 128 # hidden size of neural network

    #     fe = RecurrentFeatureExtractor(lexicon=list(range(10)),
    #                                    H=H,
    #                                    bidirectional=True)

    #     PCFG_predictor = PCFG_Predictor(
    #         fe,
    #         template_cfg=template_cfg
    #         )

    #     Q_predictor = Q_Predictor(
    #         fe,
    #         template_dsl=template_dsl,
    #         template_cfg=template_cfg
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
    #     # Q_predictor.train(programs, tasks)
    #     # Q_predictor.test(programs, tasks)

    # def test_programs(self):
    #     H = 128 # hidden size of neural network

    #     fe = RecurrentFeatureExtractor(lexicon=list(range(10)),
    #                                    H=H,
    #                                    bidirectional=True)

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
    #                                   n_gram = 1)

    #     programs = [
    #         Function(BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))),[BasicPrimitive("0", INT),BasicPrimitive("1", INT)], INT),
    #         BasicPrimitive("0", INT),
    #         BasicPrimitive("1", INT)
    #         ]

    #     # programs = [
    #     #     Function(
    #     #         BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #     #         [BasicPrimitive("1", INT)], 
    #     #         Arrow(INT, INT)),

    #     #     Function(
    #     #         BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #     #         [Function(
    #     #             BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))), 
    #     #             [BasicPrimitive("1", INT), BasicPrimitive("1", INT)],
    #     #             INT),
    #     #         ],
    #     #         Arrow(INT, INT))
    #     #     ]

    #     # x = [4] # input
    #     # y = [5] # output
    #     # ex1 = (x,y) # a single input/output example

    #     # x = [9] # input
    #     # y = [10] # output
    #     # ex2 = (x,y) # a single input/output example

    #     # task1 = [ex1,ex2] # a task is a list of input/outputs

    #     # x = [8] # input
    #     # y = [10] # output
    #     # ex1 = (x,y) # a single input/output example

    #     # task2 = [ex1] # a task is a list of input/outputs

    #     # tasks = [task1,task2]

    #     xs = ([1,9,7],[8,8,7]) # inputs
    #     y = [4] # output
    #     ex1 = (xs,y) # some input/output example

    #     xs = ([1,3,7],[1,7]) # inputs
    #     y = [6] # output
    #     ex2 = (xs,y) # another input/output example

    #     task = [ex1,ex2] # a task is a list of input/outputs
    #     task2 = [ex1]
    #     task3 = [ex2]

    #     assert fe.forward_one_task( task).shape == torch.Size([H])
    #     # batched forward pass - test cases
    #     assert fe.forward([task,task2,task3]).shape == torch.Size([3,H])
    #     assert torch.all( fe.forward([task,task2,task3])[0] == fe.forward_one_task(task) )
    #     assert torch.all( fe.forward([task,task2,task3])[1] == fe.forward_one_task(task2) )
    #     assert torch.all( fe.forward([task,task2,task3])[2] == fe.forward_one_task(task3) )

    #     # pooling of examples happens through averages - check via this assert
    #     assert(torch.stack([fe.forward_one_task(task3),fe.forward_one_task(task2)],0).mean(0) - fe.forward_one_task(task)).abs().max() < 1e-5
    #     assert(torch.stack([fe.forward_one_task(task),fe.forward_one_task(task)],0).mean(0) - fe.forward_one_task(task3)).abs().max() > 1e-5

    #     tasks = [task,task2,task3]

    #     models = {
    #     "has Q": RecognitionModel(
    #         fe, 
    #         template_dsl=template_dsl, 
    #         template_cfg=template_cfg,
    #         type_request=type_request
    #         ),
    #     "has no Q (directly predict probabilities)": RecognitionModel(
    #         fe,
    #         template_cfg=template_cfg
    #         )
    #     }

    #     for model_name, model in models.items():
    #         print("training model", model_name)
    #         optimizer = torch.optim.Adam(model.parameters())

    #         for step in range(200):
    #             optimizer.zero_grad()
    #             grammars = model(tasks)
    #             likelihood = sum(g.log_probability_program(template_cfg.start, p)
    #                              for g,p in zip(grammars, programs))
    #             (-likelihood).backward()
    #             optimizer.step()

    #             if step % 100 == 0:
    #                 print("optimization step", step, "\tlog likelihood ", likelihood)

    #         grammars = model(tasks)
    #         for g, p in zip(grammars, programs):
    #             grammar = g.normalise()
    #             print("predicted grammar", grammar)
    #             print("intended program", p)
    #             print("probability of the intended program", 
    #                 grammar.probability_program(template_cfg.start, p))

if __name__ == "__main__":
    unittest.main(verbosity=2)
