import torch
import torch.nn as nn
import numpy as np

from dsl import *
from pcfg import PCFG
from pcfg_logprob import LogProbPCFG

class Q_Predictor(nn.Module):
    def __init__(self, 
        feature_extractor,
        template_dsl,
        template_cfg,
        list_variables):
        """
        feature_extractor: a neural network module 
        taking a list of tasks (each task is a list of input-outputs) 
        and returning a tensor of shape 
        [len(list_of_tasks), feature_extractor.output_dimensionality]

        template_dsl: a dsl giving the list of primitives

        template_cfg: a cfg giving the structure that will be output 
        """
        super(Q_Predictor, self).__init__()

        self.feature_extractor = feature_extractor
        H = self.feature_extractor.output_dimensionality # hidden
      
        self.template_cfg = template_cfg
        self.template_dsl = template_dsl

        self.number_of_primitives = len(template_dsl.list_primitives)
        self.list_variables = list_variables
        self.number_of_outputs = self.number_of_primitives + len(self.list_variables)
        self.number_of_parents = self.number_of_primitives + 1 # parent can be None
        self.maximum_arguments = max(len(primitive.type.arguments())
                                     for primitive in template_dsl.list_primitives)
        self.q_predictor = nn.Linear(H,
             self.number_of_outputs*self.number_of_parents*self.maximum_arguments)

    def q_vector_to_dictionary(self,q):
        """
        q: size self.number_of_outputs*self.number_of_parents*self.maximum_arguments
        """
        q = q.view(self.number_of_parents, self.maximum_arguments, self.number_of_outputs)
        q = nn.LogSoftmax(-1)(q)

        q_dictionary = {}
        for parent_index, parent in enumerate([None] + self.template_dsl.list_primitives):
            for argument_number in range(self.maximum_arguments):
                for primitive_index, primitive in enumerate(self.template_dsl.list_primitives):
                    q_dictionary[parent, argument_number, primitive] = \
                    q[parent_index, argument_number, primitive_index]
                for variable_index, variable in enumerate(self.list_variables):
                    q_dictionary[parent, argument_number, variable] = \
                    q[parent_index, argument_number, self.number_of_primitives + variable_index]
        return q_dictionary
            
    def forward(self, tasks):
        """
        tasks: list of tasks

        returns: list of PCFGs
        """
        features = self.feature_extractor(tasks)

        q = self.q_predictor(features)
        q_dictionary = [self.q_vector_to_dictionary(q[b])
              for b in range(len(tasks))]
        grammars = [self.template_cfg.Q_to_PCFG(q_dictionary[b])
              for b in range(len(tasks))]
        return grammars

    def train(self, programs, tasks, epochs=200):
        optimizer = torch.optim.Adam(self.parameters())

        for step in range(epochs):
            optimizer.zero_grad()
            grammars = self(tasks)
            likelihood = sum(grammar.log_probability_program(self.template_cfg.start, program)
                             for grammar,program in zip(grammars, programs))
            (-likelihood).backward()
            optimizer.step()

            # if step % 100 == 0:
            #     logging.debug("optimization step {}\tlog likelihood {}".format(step, likelihood))
            #     logging.debug("grammars {}".format(grammars))

    def test(self, programs, tasks):
        grammars = self(tasks)
        for grammar, program in zip(grammars, programs):
            grammar = grammar.normalise()
            logging.debug("predicted grammar {}".format(grammar))
            logging.info("intended program {}\nprobability {}".format(program, grammar.probability_program(self.template_cfg.start, program)))
