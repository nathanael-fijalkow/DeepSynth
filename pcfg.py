import random
import numpy as np
from math import prod

import vose

from type_system import *
from program import Program, Function, Variable, BasicPrimitive, New

# make sure hash is deterministic
PYTHONHASHSEED = 0

class PCFG:
    """
    Object that represents a probabilistic context-free grammar

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary : {P : l, w}
    with P a program, l a list of non-terminals, and w a weight
    representing the derivation S -> P(S1, S2, ...) with weight w for l' = [S1, S2, ...]

    list_derivations: a dictionary of type {S: l}
    with S a non-terminal and l the list of programs P appearing in derivations from S,
    sorted from most probable to least probable

    max_probability: a dictionary of type {S: (Pmax, probability)} cup {(S, P): (Pmax, probability)}
    with S a non-terminal

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in max_probability
    """

    def __init__(self, start, rules, max_program_depth=4):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        self.hash = hash(format(rules))

        self.remove_non_productive(max_program_depth)
        self.remove_non_reachable(max_program_depth)

        for S in self.rules:
            s = sum([self.rules[S][P][1] for P in self.rules[S]])
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                self.rules[S][P] = (args_P, w / s)

        self.hash_table_programs = {}
        self.max_probability = {}
        self.compute_max_probability()

        self.list_derivations = {}
        self.vose_samplers = {}

        for S in self.rules:
            self.list_derivations[S] = sorted(
                self.rules[S], key=lambda P: self.rules[S][P][1]
            )
            self.vose_samplers[S] = vose.Sampler(
                np.array([self.rules[S][P][1] for P in self.list_derivations[S]])
            )

    def return_unique(self, P):
        """
        ensures that if a program appears in several rules,
        it is represented by the same object
        """
        if P.hash in self.hash_table_programs:
            return self.hash_table_programs[P.hash]
        else:
            self.hash_table_programs[P.hash] = P
            return P

    def remove_non_productive(self, max_program_depth=4):
        """
        remove non-terminals which do not produce programs
        """
        new_rules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                if all([arg in new_rules for arg in args_P]) and w > 0:
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

    def compute_max_probability(self):
        """
        populates the dictionary max_probability
        """
        for S in reversed(self.rules):
            best_program = None
            best_probability = 0

            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                P_unique = self.return_unique(P)

                if len(args_P) == 0:
                    self.max_probability[(S, P)] = P_unique
                    P_unique.probability[(self.__hash__(), S)] = w
                    assert P_unique.probability[
                        (self.__hash__(), S)
                    ] == self.probability_program(S, P_unique)

                else:
                    new_program = Function(
                        function=P_unique,
                        arguments=[self.max_probability[arg] for arg in args_P],
                        type_=S[0],
                        probability={},
                    )
                    P_unique = self.return_unique(new_program)
                    probability = w
                    for arg in args_P:
                        probability *= self.max_probability[arg].probability[(self.__hash__(), arg)]
                    self.max_probability[(S, P)] = P_unique
                    assert (self.__hash__(), S) not in P_unique.probability
                    P_unique.probability[(self.__hash__(), S)] = probability
                    assert probability == self.probability_program(S, P_unique)

                if (
                    self.max_probability[(S, P)].probability[(self.__hash__(), S)]
                    > best_probability
                ):
                    best_program = self.max_probability[(S, P)]
                    best_probability = self.max_probability[(S, P)].probability[
                        (self.__hash__(), S)
                    ]

            assert best_probability > 0
            self.max_probability[S] = best_program

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["vose_samplers"]
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.vose_samplers = {
            S: vose.Sampler(
                np.array([self.rules[S][P][1] for P in self.list_derivations[S]])
            )
            for S in self.rules
        }

    def __hash__(self):
        return self.hash

    def __repr__(self):
        s = "Print a PCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w)
        return s

    def sampling(self):
        """
        A generator that samples programs according to the PCFG G
        """

        while True:
            yield self.sample_program(self.start)

    def sample_program(self, S):
        i = self.vose_samplers[S].sample()
        P = self.list_derivations[S][i]
        args_P, w = self.rules[S][P]
        if len(args_P) == 0:
            return P
        arguments = []
        for arg in args_P:
            arguments.append(self.sample_program(arg))
        return Function(P, arguments)

    def probability_program(self, S, P):
        """
        Compute the probability of a program P generated from the non-terminal S
        """
        if isinstance(P, (Variable, BasicPrimitive, New)):
            return self.rules[S][P][1]
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]
            for i, arg in enumerate(args_P):
                probability *= self.probability_program(self.rules[S][F][0][i], arg)
            return probability
        assert False
