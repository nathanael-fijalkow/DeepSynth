from numpy import result_type
from math import sqrt
from collections import deque

from pcfg import PCFG

try:
    from math import prod
except:

    def prod(x):
        p = 1
        for y in x:
            p = p * y
        return y


def sqrt_sampling_with_sbsur(G: PCFG, batch_size: int = 100, start = None):
    """
    A generator that samples programs according to the sqrt of the PCFG G.
    It uses Stochastic Beam Search and Unique Randomizer to sample unique programs.
    Note that contrary to sqrt_sampling, it yields compressed programs.
    """
    SQRT = sqrt_PCFG(G)
    start = start or SQRT.start
    sampler = SQRT.get_sbsur_sampler(SQRT.start)
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
    while True:
        yield SQRT.sample_program(SQRT.start)


def sqrt_PCFG(G: PCFG):
    """
    Input: a PCFG G
    Output: a PCFG that is the sqrt of G
    """
    WCFG_rules = {}
    for S in G.rules:
        WCFG_rules[S] = {
            F: (G.rules[S][F][0], G.rules[S][F][1] ** (0.5)) for F in G.rules[S]
        }

    # Yeah, I know... not exactly a PCFG (probabilities do not sum to 1), but it fits the bill
    WCFG = PCFG(start=G.start, rules=WCFG_rules)
    partition_function = compute_partition_function(WCFG)

    PCFG_rules = {}
    for S in WCFG.rules:
        new_rules_S = {}
        for F in WCFG.rules[S]:
            args_F = WCFG.rules[S][F][0]
            w = WCFG.rules[S][F][1]
            multiplier = prod(partition_function[arg] for arg in args_F)
            new_rules_S[F] = (args_F, w * multiplier * 1 / partition_function[S])
        PCFG_rules[S] = new_rules_S
    return PCFG(G.start, PCFG_rules)


def compute_partition_function(G: PCFG):
    """
    Computes the so-called partition function Z as a dictionary {S: Z(S)}
    where Z(S) = sum_{P generated from S} Probability(P)
    """
    Z = {S: 1 for S in G.rules}
    for S in reversed(G.rules):
        s = 0
        for F in G.rules[S]:
            args_F, w = G.rules[S][F]
            prod = w
            for arg in args_F:
                prod *= Z[arg]
            s += prod
        Z[S] = s
    return Z
