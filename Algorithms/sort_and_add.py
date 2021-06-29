from pcfg import *
from Algorithms.dfs import *

def sort_and_add(G : PCFG, init = 5, step = 5):
    '''
    A generator that enumerates all programs using incremental search over a DFS 
    '''
    size = init
    print("Initialising with size {}".format(size))
    G_truncated = truncate(G, size)
    gen = dfs(G_truncated)
    
    while True:
        try:
            yield next(gen)
        except StopIteration:
            size += step
            print("Increasing size to {}".format(size))
            G_truncated = truncate(G, size)
            gen = dfs(G_truncated)

def truncate(G: PCFG, size):
    new_rules = {}
    for S in G.rules:
        new_rules[S] = G.rules[S][:size]
        s = sum(w for (_,_,w) in new_rules[S])
        new_rules[S] = [(F, args_F, w/s) for (F, args_F, w) in new_rules[S]]
    return PCFG(G.start, new_rules)