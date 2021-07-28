from collections import deque
from heapq import heappush, heappop

from pcfg import PCFG


def a_star(G: PCFG):
    """
    A generator that enumerates all programs using A*.
    Assumes that the PCFG only generates programs of bounded depth.
    """

    ## compute max probability
    G.compute_max_probability()

    frontier = []
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    heappush(
        frontier,
        (
            -G.max_probability[G.start].probability[(G.__hash__(), G.start)],
            (None, initial_non_terminals, 1),
        ),
    )
    # A frontier is a heap of pairs (-max_probability, (partial_program, non_terminals, probability))
    # describing a partial program:
    # max_probability is the most likely program completing the partial program
    # partial_program is the list of primitives and variables describing the leftmost derivation,
    # non_terminals is the queue of non-terminals appearing from left to right, and
    # probability is the probability of the partial program

    while len(frontier) != 0:
        max_probability, (partial_program, non_terminals, probability) = heappop(
            frontier
        )
        if len(non_terminals) == 0:
            yield partial_program
        else:
            S = non_terminals.pop()
            for P in G.rules[S]:
                args_P, w = G.rules[S][P]
                new_partial_program = (P, partial_program)
                new_non_terminals = non_terminals.copy()
                new_probability = probability * w
                new_max_probability = new_probability
                for arg in args_P:
                    new_non_terminals.append(arg)
                    new_max_probability *= G.max_probability[arg].probability[
                        (G.__hash__(), arg)
                    ]
                heappush(
                    frontier,
                    (
                        -new_max_probability,
                        (new_partial_program, new_non_terminals, new_probability),
                    ),
                )
