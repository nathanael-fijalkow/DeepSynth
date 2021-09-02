from math import sqrt
from collections import deque

from pcfg import PCFG

def sqrt_sampling_with_sbsur(G: PCFG, batch_size: int = 100, start = None):
    """
    A generator that samples programs according to the sqrt of the PCFG G.
    It uses Stochastic Beam Search and Unique Randomizer to sample unique programs.
    Note that contrary to sqrt_sampling, it yields compressed programs.
    """
    try:
        G.list_derivations
    except:
        G.list_derivations = {}
        for S in G.rules:
            G.list_derivations[S] = sorted(
                G.rules[S], key=lambda P: G.rules[S][P][1]
            )
    SQRT = sqrt_PCFG(G)
    start = start or SQRT.start
    sampler = SQRT.get_sbsur_sampler(start)
    while True:
        batch = sampler(batch_size)
        for el in batch:
            yield el
        if len(batch) < batch_size:
            break

def sqrt_sampling(G: PCFG):
    """
    A generator that samples programs according to the sqrt of the PCFG G
    """
    SQRT = sqrt_PCFG(G)
    SQRT.init_vose()

    while True:
        yield SQRT.sample_program(SQRT.start)

def sqrt_PCFG(G: PCFG):
    """
    Input: a PCFG G
    Output: a PCFG that is the sqrt of G
    """
    Z = {}
    # Z[S] = sum_{prog generated from S} Prob(prog)
    # Z[S,P] = sum_{prog generated from S using rule P} Prob(prog)
    for S in reversed(G.rules):
        s = 0
        for P in G.rules[S]:
            args_P, w = G.rules[S][P]
            prob = sqrt(w) 
            for arg in args_P:
                prob *= Z[arg]
            Z[S,P] = prob
            s += prob
        Z[S] = s

    new_rules = {}
    for S in G.rules:
        new_rules[S] = {}
        for P in G.rules[S]:
            args_P, w = G.rules[S][P]
            new_rules[S][P] = args_P, Z[S,P] / Z[S]
    return PCFG(G.start, new_rules, G.max_program_depth)
