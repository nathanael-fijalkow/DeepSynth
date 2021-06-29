from program import *
from pcfg import *

from collections import deque 
from heapq import heappush, heappop, heappushpop

def bfs(G : PCFG, beam_width = 50000):
    '''
    A generator that enumerates all programs using a BFS.
    Assumes that the PCFG only generates programs of bounded depth.
    '''
    # We reverse the rules: they should be non-decreasing
    for S in G.rules:
        sorted(G.rules[S], key=lambda x: G.rules[S][x][1])

    frontier = []
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    heappush(frontier, (1, (None, initial_non_terminals)))
    # A frontier is a heap of pairs (probability, (partial_program, non_terminals)) 
    # describing a partial program:
    # partial_program is the list of primitives and variables describing the leftmost derivation, and
    # non_terminals is the queue of non-terminals appearing from left to right
    # probability is the probability

    while True:
        new_frontier = []
        while True:
            try:
                probability, (partial_program, non_terminals) = heappop(frontier)
                if len(non_terminals) == 0: 
                    yield partial_program
                else:
                    S = non_terminals.pop()
                    for P in G.rules[S]:
                        args_P, w = G.rules[S][P]
                        new_partial_program = (P, partial_program)
                        new_non_terminals = non_terminals.copy()
                        new_probability = probability * w
                        for arg in args_P:
                            new_non_terminals.append(arg)
                        if len(new_frontier) <= beam_width:
                            heappush(new_frontier, (new_probability, (new_partial_program, new_non_terminals)))
                        else:
                            heappushpop(new_frontier, (new_probability, (new_partial_program, new_non_terminals)))
            except IndexError:
                frontier = new_frontier
                break
