
import typing
import tqdm
from pcfg import PCFG
import logging
from program import Program
import time
from typing import Callable, Tuple

from Algorithms.heap_search import heap_search
from Algorithms.heap_search_naive import heap_search_naive
from Algorithms.a_star import a_star
from Algorithms.threshold_search import threshold_search
from Algorithms.dfs import dfs
from Algorithms.bfs import bfs
from Algorithms.sort_and_add import sort_and_add
from Algorithms.sqrt_sampling import sqrt_sampling, sqrt_sampling_with_sbsur

from program_as_list import reconstruct_from_compressed

logging_levels = {0: logging.INFO, 1: logging.DEBUG}


verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])
timeout = 100
total_number_programs = 1_000_000

list_algorithms = [
    (heap_search, 'heap search', {}),
    (heap_search_naive, 'heap search naive', {}),
    (sqrt_sampling, 'SQRT', {}),
    (sqrt_sampling_with_sbsur, 'SQRT+SBS UR', {}),
    (a_star, 'A*', {}),
    (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}),
    (bfs, 'bfs', {'beam_width' : 50000}),
    (dfs, 'dfs', {}),
    (sort_and_add, 'sort and add', {}),
]
# Set of algorithms where we need to reconstruct the programs
reconstruct = {dfs, bfs, threshold_search, a_star,
               sort_and_add, sqrt_sampling_with_sbsur}


def run_algorithm(is_correct_program: Callable[[Program], bool], pcfg: PCFG, algo_index: int) -> Tuple[Program, float, float, int, float, float]:
    '''
    Run the algorithm until either timeout or 1M programs, and for each program record probability and time of output
    return program, search_time, evaluation_time, nb_programs, cumulative_probability, probability
    '''
    algorithm, name_algo, param = list_algorithms[algo_index]
    search_time = 0
    evaluation_time = 0
    gen = algorithm(pcfg, **param)
    found = False
    if name_algo == "SQRT":
        _ = next(gen)
    nb_programs = 0
    cumulative_probability = 0

    while (search_time + evaluation_time < timeout and nb_programs < total_number_programs):

        # Searching for the next program
        search_time -= time.perf_counter()
        try:
            program = next(gen)
        except:
            search_time += time.perf_counter()
            logging.debug(
                "Output the last program after {}".format(nb_programs))
            break  # no next program

        # Reconstruction if needed
        if algorithm in reconstruct:
            target_type = pcfg.start[0]
            program = reconstruct_from_compressed(program, target_type)
        search_time += time.perf_counter()
        logging.debug('program found: {}'.format(program))

        if program == None:
            logging.debug(
                "Output the last program after {}".format(nb_programs))
            break

        nb_programs += 1
        probability = pcfg.probability_program(pcfg.start, program)
        cumulative_probability += probability
        logging.debug('probability: %s' %
                      probability)

        # Evaluation of the program
        evaluation_time -= time.perf_counter()
        found = is_correct_program(program)
        evaluation_time += time.perf_counter()

        if nb_programs % 100_000 == 0:
            logging.debug('tested {} programs'.format(nb_programs))

        if found:
            logging.debug("\nSolution found: %s" % program)
            logging.debug('[NUMBER OF PROGRAMS]: %s' % nb_programs)
            logging.debug("[SEARCH TIME]: %s" % search_time)
            logging.debug("[EVALUATION TIME]: %s" % evaluation_time)
            logging.debug("[TOTAL TIME]: %s" % (evaluation_time + search_time))
            return program, search_time, evaluation_time, nb_programs, cumulative_probability, probability

    logging.debug("\nNot found")
    logging.debug('[NUMBER OF PROGRAMS]: %s' % nb_programs)
    logging.debug("[SEARCH TIME]: %s" % search_time)
    logging.debug("[EVALUATION TIME]: %s" % evaluation_time)
    logging.debug("[TOTAL TIME]: %s" % (evaluation_time + search_time))
    return None, timeout, timeout, nb_programs, cumulative_probability, probability


def gather_data(dataset: typing.List[Tuple[str, PCFG, Callable]], algo_index: int) -> typing.List[Tuple[str, Tuple[Program, float, float, int, float, float]]]:
    algorithm, _, _ = list_algorithms[algo_index]
    logging.info('\n## Running: %s' % algorithm.__name__)
    output = []

    for task_name, pcfg, is_correct_program in tqdm.tqdm(dataset):
        logging.debug("## Task:", task_name)
        data = run_algorithm(is_correct_program, pcfg, algo_index)
        output.append((task_name, data))
    return output
