from pcfg import *
from Algorithms.dfs import *

import logging

def sort_and_add(G : PCFG, init = 5, step = 5):
    '''
    A generator that enumerates all programs using incremental search over a DFS 
    '''
    size = init
    logging.debug("Initialising with size {}".format(size))

    try:
        G.list_derivations
    except:
        G.list_derivations = {}
        for S in G.rules:
            G.list_derivations[S] = sorted(
                G.rules[S], key=lambda P: G.rules[S][P][1]
            )
    G_truncated = truncate(G, size)
    gen = dfs(G_truncated)
    
    while True:
        try:
            yield next(gen)
        except StopIteration:
            size += step
            logging.debug("Increasing size to {}".format(size))
            G_truncated = truncate(G, size)
            gen = dfs(G_truncated)

def truncate(G: PCFG, size):
    new_rules = {}
    for S in G.rules:
        new_rules[S] = {}
        s = sum(G.rules[S][P][1] for P in G.list_derivations[S][:size])
        for i, P in enumerate(G.list_derivations[S]):
            if i < size:
                args_P, w = G.rules[S][P]
                new_rules[S][P] = args_P, w / s
    return PCFG(G.start, new_rules, max_program_depth = G.max_program_depth)
