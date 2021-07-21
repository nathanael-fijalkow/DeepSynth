import torch
import torch.nn as nn
import numpy as np

from dsl import *
from pcfg import PCFG
from pcfg_logprob import LogProbPCFG

class PCFG_Predictor(nn.Module):
    def __init__(self, 
        feature_extractor,
        template_cfg):
        """
        feature_extractor: a neural network module 
        taking a list of tasks (each task is a list of input-outputs) 
        and returning a tensor of shape 
        [len(list_of_tasks), feature_extractor.output_dimensionality]

        template_cfg: a cfg giving the structure that will be output 
        """
        super(PCFG_Predictor, self).__init__()

        self.feature_extractor = feature_extractor 
        self.template_cfg = template_cfg
        H = self.feature_extractor.output_dimensionality # hidden

        projection_layer = {}
        for S in template_cfg.rules:
            n_productions = len(template_cfg.rules[S])
            module = nn.Sequential(nn.Linear(H, n_productions),
                                   nn.LogSoftmax(-1))
            projection_layer[str(S)] = module
        self.projection_layer = nn.ModuleDict(projection_layer)
            
    def forward(self, tasks):
        """
        tasks: list of tasks

        returns: list of PCFGs
        """
        features = self.feature_extractor(tasks)
        template_cfg = self.template_cfg

        probabilities = {S: self.projection_layer[str(S)](features)
                         for S in template_cfg.rules}
        grammars = []
        for b in range(len(tasks)): # iterate over batches
            rules = {}
            for S in template_cfg.rules:
                rules[S] = {}
                for i, P in enumerate(template_cfg.rules[S]):
                    rules[S][P] = template_cfg.rules[S][P], probabilities[S][b, i]
            grammar = LogProbPCFG(template_cfg.start, 
                rules, 
                max_program_depth=template_cfg.max_program_depth)
            grammars.append(grammar)
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

            if step % 100 == 0:
                logging.debug("optimization step {}\tlog likelihood {}".format(step, likelihood))

    def test(self, programs, tasks):
        grammars = self(tasks)
        for grammar, program in zip(grammars, programs):
            grammar = grammar.normalise()
            logging.debug("predicted grammar {}".format(grammar))
            logging.info("intended program {}\nprobability {}".format(program, grammar.probability_program(self.template_cfg.start, program)))
