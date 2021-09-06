from program import *
from pcfg import *

from collections import deque 
from heapq import heappush, heappop, heappushpop

def bfs(G : PCFG, beam_width = 50_000):
    '''
    A generator that enumerates all programs using a BFS.
    '''
    frontier = []
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    frontier.append((None, initial_non_terminals))
    # A frontier is a list of programs ((partial_program, non_terminals)) 
    # describing a partial program:
    # partial_program is the list of primitives and variables describing the leftmost derivation, and
    # non_terminals is the queue of non-terminals appearing from left to right

    while True:
        new_frontier = []
        if len(frontier) == 0:
            raise StopIteration
        while True:
            try:
                (partial_program, non_terminals) = frontier.pop()
                if len(non_terminals) == 0: 
                    yield partial_program
                else:
                    S = non_terminals.pop()
                    for P in G.rules[S]:
                        args_P, w = G.rules[S][P]
                        new_partial_program = (P, partial_program)
                        new_non_terminals = non_terminals.copy()
                        for arg in args_P:
                            new_non_terminals.append(arg)
                        if len(new_frontier) <= beam_width:
                            new_frontier.append((new_partial_program, new_non_terminals))
            except IndexError:
                frontier = new_frontier
                break
