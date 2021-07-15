from type_system import *
from program import *

class CFG:
    '''
    Object that represents a context-free grammar
 
    start: a non-terminal

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary {P : l} with P a program 
    and l a list of non-terminals representing the derivation S -> P(S1,S2,..) 
    with l = [S1,S2,...]

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in rules

    '''
    def __init__(self, start, rules, max_program_depth):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        self.remove_non_productive(max_program_depth)
        self.remove_non_reachable(max_program_depth)
            
        # checks that all non-terminals are productive
        for S in self.rules:
            # print("\n\n###########\nLooking at S", S)            
            assert(len(self.rules[S]) > 0)
            for P in self.rules[S]:
                args_P = self.rules[S][P]
                # print("####\nFrom S: ", S, "\nargument P: ", P, args_P)
                for arg in args_P:
                    # print("checking", arg)
                    assert(arg in self.rules)

    def remove_non_productive(self, max_program_depth = 4):
        '''
        remove non-terminals which do not produce programs
        '''
        new_rules = {}
        for S in reversed(self.rules):
            # print("\n\n###########\nLooking at S", S)            
            for P in self.rules[S]:
                args_P = self.rules[S][P]
                # print("####\nFrom S: ", S, "\nargument P: ", P, args_P)
                if all([arg in new_rules for arg in args_P]):
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]
                # else:
                #     print("the rule {} from {} is non-productive".format(P,S))

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]
                # print("the non-terminal {} is non-productive".format(S))

    def remove_non_reachable(self, max_program_depth = 4):
        '''
        remove non-terminals which are not reachable from the initial non-terminal
        '''
        reachable = set()
        reachable.add(self.start)

        reach = set()
        new_reach = set()
        reach.add(self.start)

        for i in range(max_program_depth):
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
                # print("the non-terminal {} is not reachable:".format(S))

    def __repr__(self):
        s = "Print a CFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += '#\n {}\n'.format(S)
            for P in self.rules[S]:
                s += '   {} - {}: {}\n'.format(P, P.type, self.rules[S][P])
        return s
