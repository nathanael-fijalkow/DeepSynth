import random
import numpy as np
from math import exp

import vose

from type_system import *
from program import Program, Function, Variable, BasicPrimitive, New
from pcfg import PCFG

# make sure hash is deterministic
PYTHONHASHSEED = 0

class LogProbPCFG:
    """
    Object that represents a probabilistic context-free grammar

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary : {P : l, w}
    with P a program, l a list of non-terminals, and w a weight
    representing the derivation S -> P(S1, S2, ...) with weight w for l' = [S1, S2, ...]
    """

    def __init__(self, start, rules, max_program_depth=4):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        # self.hash = hash(format(rules))
        # self.remove_non_productive(max_program_depth)
        # self.remove_non_reachable(max_program_depth)

    def remove_non_productive(self, max_program_depth=4):
        """
        remove non-terminals which do not produce programs
        """
        new_rules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                if all([arg in new_rules for arg in args_P]):
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]

    def remove_non_reachable(self, max_program_depth=4):
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable = set()
        reachable.add(self.start)

        reach = set()
        new_reach = set()
        reach.add(self.start)

        for i in range(max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P, _ = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def __hash__(self):
        return self.hash

    def __repr__(self):
        s = "Print a LogProbPCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w)
        return s

    def log_probability_program(self, S, P):
        """
        Compute the log probability of a program P generated from the non-terminal S
        """
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]
            for i, arg in enumerate(args_P):
                probability = probability + self.log_probability_program(self.rules[S][F][0][i], arg)
            return probability

        if isinstance(P, (Variable, BasicPrimitive, New)):
            return self.rules[S][P][1]
        assert False

    def normalise(self):
        self.remove_non_productive(self.max_program_depth)
        self.remove_non_reachable(self.max_program_depth)
        normalised_rules = {}
        for S in self.rules:
            s = sum(exp(self.rules[S][P][1].item()) for P in self.rules[S])
            if s > 0:
                normalised_rules[S] = {}
                for P in self.rules[S]:
                    normalised_rules[S][P] = \
                    self.rules[S][P][0], exp(self.rules[S][P][1].item()) / s
        return PCFG(self.start, 
            normalised_rules, 
            max_program_depth=self.max_program_depth)
    
