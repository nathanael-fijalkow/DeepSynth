from program import *
from pcfg import *
from Algorithms.sqrt_sampling import *
from Algorithms.parallel import parallel_workers

from collections import deque 

def hybrid(G : PCFG, DFS_depth = 3, width = 20, batch_size = 100000, CPUs=1, timeout=5):
    '''
    A generator that enumerates all programs using a hybrid BFS + SQRT sampling.
    '''

    SQRT = sqrt_PCFG(G)

    frontier = []
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    frontier.append((None, initial_non_terminals, 1))
    # A frontier is a list of triples (partial_program, non_terminals, probability) 
    # describing a partial program:
    # partial_program is the list of primitives and variables describing the leftmost derivation, and
    # non_terminals is the queue of non-terminals appearing from left to right
    # probability is the probability

    for d in range(DFS_depth):
        new_frontier = []
        while True:
            try:
                (partial_program, non_terminals, probability) = frontier.pop()
                if len(non_terminals) > 0: 
                    S = non_terminals.pop()
                    for F, args_F, w in G.rules[S][:width]:
                        new_partial_program = (F, partial_program)
                        new_non_terminals = non_terminals.copy()
                        for arg in args_F:
                            new_non_terminals.append(arg)
                        new_probability = probability * w
                        new_frontier.append((new_partial_program, new_non_terminals, new_probability))
            except IndexError:
                frontier = new_frontier
                break

    # print("The DFS phase generated {} programs".format(len(frontier)))
    list_programs = []
    # set_non_terminals = set()
    for (partial_program, non_terminals, probability) in frontier:
        program_as_list = []
        list_from_compressed(partial_program, program_as_list)
        weight = int(batch_size * probability)
        # print("the weight for {} is {}".format(program_as_list, weight))
        for i in range(weight):
            list_programs.append((program_as_list, non_terminals, probability))
        # if weight > 0:
        #     for S in non_terminals:
        #         set_non_terminals.add(S)
    # print("We now have a list of {} programs with repetitions".format(len(list_programs)))
    # print("We now have a list of {} programs with repetitions using a total of {} non-terminals".format(len(list_programs), len(set_non_terminals)))

    if CPUs == 1: # no parallelism
        while True:
            # Idea: do we want to re-use sampled programs in non-terminals where they fit?
            for (partial_program, non_terminals, probability) in list_programs:
                new_program = partial_program.copy()
                for S in non_terminals:
                    new_program += SQRT.sample_program(S)
                yield new_program

    else:
        # TODO: workers need to have the timeout properly set
        # TODO: workers need to check whether the sampled program is correct;
        #       otherwise, you end up with way too much communication between processes
        yield from parallel_workers(CPUs,
                                    [(lp,timeout) for lp in list_programs],
                                    parallel_callback)

def parallel_callback(list_program, timeout):
    start_time = time.time()
    
    (partial_program, non_terminals, probability) = list_program
    while time.time() < time.time() + timeout:
        new_program = partial_program.copy()
        for S in non_terminals:
            new_program += SQRT.sample_program(S)
            yield new_program

def list_from_compressed(program, program_as_list = []):
    (F, sub_program) = program
    if sub_program:
        list_from_compressed(sub_program, program_as_list)
    program_as_list.append(F)



