import random
import numpy as np

from pcfg_logprob import LogProbPCFG
from pcfg import PCFG

class CFG:
    '''
    Object that represents a context-free grammar with normalised probabilites
 
    start: a non-terminal

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary {P : l} with P a program 
    and l a list of non-terminals representing the derivation S -> P(S1,S2,..) 
    with l = [S1,S2,...]

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in rules

    '''
    def __init__(self, start, rules, max_program_depth, clean=True):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        if clean:
            self.remove_non_productive()
            self.remove_non_reachable()
            
    def remove_non_productive(self):
        '''
        remove non-terminals which do not produce programs
        '''
        new_rules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P = self.rules[S][P]
                if all(arg in new_rules for arg in args_P):
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]

    def remove_non_reachable(self):
        '''
        remove non-terminals which are not reachable from the initial non-terminal
        '''
        reachable = set()
        reachable.add(self.start)

        reach = set()
        new_reach = set()
        reach.add(self.start)

        for i in range(self.max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def __str__(self):
        s = "Print a CFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += '#\n {}\n'.format(S)
            for P in self.rules[S]:
                s += '   {} - {}: {}\n'.format(P, P.type, self.rules[S][P])
        return s

    def Q_to_LogProbPCFG(self, Q):
        rules = {}
        for S in self.rules:
            rules[S] = {}
            (_,context, _) = S
            if context:
                (old_primitive, argument_number) = context
            else:
                (old_primitive, argument_number) = None, 0
            for P in self.rules[S]:
                rules[S][P] = \
                self.rules[S][P], Q[old_primitive, argument_number, P]

        # logging.debug('Rules of the CFG from the initial non-terminal:\n%s'%str(rules[self.start]))

        return LogProbPCFG(start = self.start, 
                    rules = rules,
                    max_program_depth = self.max_program_depth)

    def CFG_to_Uniform_PCFG(self):
        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                augmented_rules[S][P] = (self.rules[S][P], 1 / p)
        return PCFG(start = self.start, 
            rules = augmented_rules, 
            max_program_depth = self.max_program_depth,
            clean = True)

    def CFG_to_Random_PCFG(self,alpha=0.7):
        new_rules = {}
        for S in self.rules:
            out_degree = len(self.rules[S])
            # weights with alpha-exponential decrease
            weights = [random.random() * (alpha ** i) for i in range(out_degree)]
            s = sum(weights)
            # normalization
            weights = [w / s for w in weights]
            random_permutation = list(
                np.random.permutation([i for i in range(out_degree)])
            )
            new_rules[S] = {}
            for i, P in enumerate(self.rules[S]):
                new_rules[S][P] = (self.rules[S][P], weights[random_permutation[i]])
        return PCFG(start = self.start, 
            rules = new_rules, 
            max_program_depth = self.max_program_depth,
            clean = True)
