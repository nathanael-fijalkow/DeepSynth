from collections import deque
from heapq import heappush, heappop

from program import Program, Function, Variable
from pcfg import PCFG

def heap_search(G: PCFG):
    H = heap_search_object(G)
    return H.generator()

class heap_search_object:
    def return_unique(self, P):
        """
        ensures that if a program appears in several heaps,
        it is represented by the same object,
        so we do not evaluate it several times
        """
        hash_P = P.hash
        if hash_P in self.hash_table_global:
            return self.hash_table_global[hash_P]
        else:
            self.hash_table_global[hash_P] = P
            return P

    def __init__(self, G: PCFG):
        self.current = None

        self.G = G
        self.start = G.start
        self.rules = G.rules
        self.symbols = [S for S in self.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps = {S: [] for S in self.symbols}

        # the same program can be pushed in different heaps, with different probabilities
        # however, the same program cannot be pushed twice in the same heap

        # self.succ[S][P] is the successor of P from S
        self.succ = {S: {} for S in self.symbols}
        # self.pred[S][P] is the hash of the predecessor of P from S
        self.pred = {S: {} for S in self.symbols}

        # self.hash_table_program[S] is the set of hashes of programs
        # ever added to the heap for S
        self.hash_table_program = {S: set() for S in self.symbols}

        # self.hash_table_global[hash] = P maps
        # hashes to programs for all programs ever added to some heap
        self.hash_table_global = {}

        # Initialisation heaps

        ## 0. compute max probability
        self.G.compute_max_probability()

        ## 1. add P(max(S1),max(S2), ...) to self.heaps[S] for all S -> P(S1, S2, ...)
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                program = self.G.max_probability[(S, P)]
                hash_program = program.hash

                # Remark: the program cannot already be in self.heaps[S]
                assert hash_program not in self.hash_table_program[S]

                self.hash_table_program[S].add(hash_program)

                # we assume that the programs from max_probability
                # are represented by the same object
                self.hash_table_global[hash_program] = program

                # print("adding to the heap", program, program.probability[S])
                heappush(
                    self.heaps[S],
                    (-program.probability[(self.G.hash, S)], program),
                )

        # 2. call query(S, None) for all non-terminal symbols S, from leaves to root
        for S in reversed(self.rules):
            self.query(S, None)

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        In other words, other will no longer be generated through heap search
        """
        our_hash = hash(other)
        self.hash_table_global[our_hash] = representative
        for S in self.G.rules:
            if our_hash in self.pred[S]:
                pred_hash = self.pred[S][our_hash]
                nxt = self.succ[S][our_hash]
                self.succ[S][pred_hash] = nxt
                self.pred[S][hash(nxt)] = pred_hash

    def generator(self):
        """
        A generator which outputs the next most probable program
        """
        while True:
            program = self.query(self.start, self.current)
            self.current = program
            yield program

    def query(self, S, program):
        """
        computing the successor of program from S
        """
        if program:
            hash_program = program.hash
        else:
            hash_program = 123891

        # if we have already computed the successor of program from S, we return its stored value
        if hash_program in self.succ[S]:
            # print("already computed the successor of S, it's ", S, program, self.succ[S][hash_program])
            return self.succ[S][hash_program]

        # otherwise the successor is the next element in the heap
        try:
            _, succ = heappop(self.heaps[S])
            # print("found succ in the heap", S, program, succ)
        except:
            return  # the heap is empty: there are no successors from S

        self.succ[S][hash_program] = succ  # we store the succesor
        self.pred[S][hash(succ)] = hash_program  # we store the predecessor

        # now we need to add all potential successors of succ in heaps[S]
        if isinstance(succ, Function):
            F = succ.function

            for i in range(len(succ.arguments)):
                # non-terminal symbol used to derive the i-th argument
                S2 = self.G.rules[S][F][0][i]
                succ_sub_program = self.query(S2, succ.arguments[i])

                if isinstance(succ_sub_program, Program):
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program

                    new_program = Function(
                        F, new_arguments, type_=succ.type, probability={}
                    )
                    new_program = self.return_unique(new_program)
                    hash_new_program = new_program.hash

                    if hash_new_program not in self.hash_table_program[S]:
                        self.hash_table_program[S].add(hash_new_program)
                        probability = self.G.rules[S][F][1]
                        for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                            probability *= arg.probability[(self.G.hash, S3)]
                        heappush(self.heaps[S], (-probability, new_program))
                        new_program.probability[(self.G.hash, S)] = probability

        if isinstance(succ, Variable):
            return succ  # if succ is a variable, there is no successor so we stop here

        return succ
