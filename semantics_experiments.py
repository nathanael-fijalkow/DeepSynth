from collections import deque
import pickle
from math import exp
import logging
import argparse
import time

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from program import Program, Function, Variable, BasicPrimitive, New
from cfg import CFG
from pcfg import PCFG
from dsl import DSL
from program_as_list import reconstruct_from_compressed

from Algorithms.heap_search import heap_search
from Algorithms.heap_search_naive import heap_search_naive
from Algorithms.a_star import a_star
from Algorithms.threshold_search import threshold_search
from Algorithms.dfs import dfs
from Algorithms.bfs import bfs
from Algorithms.sort_and_add import sort_and_add
from Algorithms.sqrt_sampling import sqrt_sampling, sqrt_sampling_with_sbsur


logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
parser.add_argument('--range_task_begin', '-rb', dest='range_task_begin', default=0)
parser.add_argument('--range_task_end', '-re', dest='range_task_end', default=1)
parser.add_argument('--timeout', '-t', dest='timeout', default=100)
parser.add_argument('--total_number_programs', '-T', dest='total_number_programs', default=1_000_000)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

timeout = int(args.timeout)
total_number_programs = int(args.total_number_programs)
range_task = range(int(args.range_task_begin), int(args.range_task_end))

# Set of algorithms where we need to reconstruct the programs
reconstruct = {dfs, bfs, threshold_search, a_star, sort_and_add, sqrt_sampling_with_sbsur}

def run_algorithm(dsl, examples, pcfg, algorithm, name_algo, param):
    '''
    Run the algorithm until either timeout or 1M programs, and for each program record probability and time of output
    '''
    logging.info('\n## Running: %s'%algorithm.__name__)
    search_time = 0
    evaluation_time = 0
    gen = algorithm(pcfg, **param)
    found = False
    if name_algo == "SQRT":
        _ = next(gen)  
        logging.debug('SQRT initialised')
    nb_programs = 0

    while (search_time + evaluation_time < timeout and nb_programs < total_number_programs):

        # Searching for the next program
        search_time -= time.perf_counter()
        try:
            program = next(gen)
        except:
            search_time += time.perf_counter()
            logging.info("Output the last program after {}".format(nb_programs))
            break # no next program            

        # Reconstruction if needed
        if algorithm in reconstruct:
            target_type = pcfg.start[0]
            program = reconstruct_from_compressed(program, target_type)
        search_time += time.perf_counter()
        logging.debug('program found: {}'.format(program))

        if program == None:
            logging.info("Output the last program after {}".format(nb_programs))
            break

        nb_programs += 1
        logging.debug('probability: %s'%pcfg.probability_program(pcfg.start, program))

        # Evaluation of the program
        evaluation_time -= time.perf_counter()
        correct = True
        i = 0
        while correct and i < len(examples):
            input_,output = examples[i]
            correct = program.eval(dsl, input_, i) == output
            i += 1
        if correct:
            found = True
        evaluation_time += time.perf_counter()

        if nb_programs % 100_000 == 0:
            logging.info('tested {} programs'.format(nb_programs))

        if found:
            logging.info("\nSolution found: %s"%program)
            logging.info('[NUMBER OF PROGRAMS]: %s'%nb_programs)
            logging.info("[SEARCH TIME]: %s"%search_time)
            logging.info("[EVALUATION TIME]: %s"%evaluation_time)
            logging.info("[TOTAL TIME]: %s"%(evaluation_time + search_time))
            return program, search_time, evaluation_time, nb_programs

    logging.info("\nNot found")
    logging.info('[NUMBER OF PROGRAMS]: %s'%nb_programs)
    logging.info("[SEARCH TIME]: %s"%search_time)
    logging.info("[EVALUATION TIME]: %s"%evaluation_time)
    logging.info("[TOTAL TIME]: %s"%(evaluation_time + search_time))
    return None, timeout, timeout, nb_programs

list_algorithms = [
    (heap_search, 'heap search', {}), 
    # (heap_search_naive, 'heap search naive', {}), 
    # (sqrt_sampling, 'SQRT', {}),
    # (sqrt_sampling_with_sbsur, 'SQRT+SBS UR', {}),
    # (a_star, 'A*', {}),
    # (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}), 
    # (bfs, 'bfs', {'beam_width' : 50000}),
    # (dfs, 'dfs', {}), 
    # (sort_and_add, 'sort and add', {}), 
    ]

for i in range_task:
    result = {}

    with open(r'tmp/list_{}.pickle'.format(i), 'rb') as f:
        name_task, dsl, pcfg, examples = pickle.load(f)

    logging.info('\n####### Solving task number {} called {}:'.format(i, name_task))
    logging.debug('Set of examples:\n %s'%examples)
    logging.debug('PCFG: %s'%str(pcfg))
    # logging.debug('PCFG: %s'%str(pcfg.rules[pcfg.start]))
    for algo, name_algo, param in list_algorithms:

        program, search_time, evaluation_time, nb_programs = run_algorithm(dsl, examples, pcfg, algo, name_algo, param)
        result[name_algo] = (name_task, search_time, evaluation_time, nb_programs)

        with open('results_semantics/semantics_experiments_{}.pickle'.format(i), 'wb') as f:
            pickle.dump(result, f)

    result.clear()
