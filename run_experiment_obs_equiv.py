import typing
import tqdm
from pcfg import PCFG
import logging
from program import BasicPrimitive, Function, New, Program, Variable
import time
from typing import Any, Callable, Tuple

from Algorithms.heap_search import heap_search


logging_levels = {0: logging.INFO, 1: logging.DEBUG}


verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])
timeout = 100
total_number_programs = 100_000_000
# Set to False to disable bottom cached evaluation for heap search
use_heap_search_cached_eval = True 

list_algorithms = [
    (heap_search, 'Heap Search', {}),
]


def run_algorithm(is_correct_program: Callable[[Program, bool], Tuple[bool, bool, Any]], pcfg: PCFG, algo_index: int) -> Tuple[Program, float, float, int, float, float, float]:
    '''
    Run the algorithm until either timeout or 1M programs, and for each program record probability and time of output
    return program, search_time, evaluation_time, nb_programs, cumulative_probability, probability
    '''
    algorithm, name_algo, param = list_algorithms[algo_index]
    search_time = 0
    merge_time = 0
    evaluation_time = 0
    gen = algorithm(pcfg, **param)
    found = False
    nb_programs = 0
    cumulative_probability = 0

    merged = 0
    fingerpints = 0

    prints = {}

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

  
        search_time += time.perf_counter()
        # logging.debug('program found: {}'.format(program))

        if program == None:
            logging.debug(
                "Output the last program after {}".format(nb_programs))
            break

        nb_programs += 1
        probability = pcfg.probability_program(pcfg.start, program)
        program_r = program

        cumulative_probability += probability
        # logging.debug('probability: %s' %
        #               probability)
        # Evaluation of the program
        evaluation_time -= time.perf_counter()
        found, is_fingerprint, fingerprint = is_correct_program(program_r, True)
        evaluation_time += time.perf_counter()

        merge_time -= time.perf_counter()
        if is_fingerprint:
            fingerpints += 1
            repr = prints.get(fingerprint, None)
            if repr is None:
                prints[fingerprint] = program_r
            else:
                gen.merge_program(repr, program_r)
                merged += 1
        merge_time += time.perf_counter()

        # if not isinstance(found, bool):
        #     found, program = found

        if nb_programs % 100_000 == 0:
            logging.debug('tested {} programs'.format(nb_programs))

        if found:
            # print("\tprogram found=", program_r)
            # logging.debug("\nSolution found: %s" % program)
            # logging.debug('[NUMBER OF PROGRAMS]: %s' % nb_programs)
            # logging.debug("[SEARCH TIME]: %s" % search_time)
            # logging.debug("[EVALUATION TIME]: %s" % evaluation_time)
            # logging.debug("[TOTAL TIME]: %s" % (evaluation_time + search_time))
            return program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability, merge_time, merged, fingerpints

    # logging.debug("\nNot found")
    # logging.debug('[NUMBER OF PROGRAMS]: %s' % nb_programs)
    # logging.debug("[SEARCH TIME]: %s" % search_time)
    # logging.debug("[EVALUATION TIME]: %s" % evaluation_time)
    # logging.debug("[TOTAL TIME]: %s" % (evaluation_time + search_time))
    # print("\tratio s/(s+e)=", search_time / (search_time + evaluation_time))
    # print("\tNot found after", nb_programs, "programs\n\tcumulative probability=",
        #   cumulative_probability, "\n\tlast probability=", probability)
    return None, search_time, evaluation_time, nb_programs, cumulative_probability, probability, merge_time, merged, fingerpints



def gather_data(dataset: typing.List[Tuple[str, PCFG, Callable]], algo_index: int) -> typing.List[Tuple[str, Tuple[Program, float, float, int, float, float]]]:
    algorithm, _, _ = list_algorithms[algo_index]
    logging.info('\n## Running: %s' % algorithm.__name__)
    output = []
    successes = 0
    pbar = tqdm.tqdm(total=len(dataset))
    pbar.set_postfix_str(f"{successes} solved")
    for task_name, pcfg, is_correct_program in dataset:
        logging.debug("## Task:", task_name)
        data = run_algorithm(is_correct_program, pcfg, algo_index)
        if not data[0]:
            logging.debug("\tsolution=", task_name)
            logging.debug("\ttype request=", pcfg.type_request())
        if isinstance(task_name, Program):
            try:
                prob = pcfg.probability_program(pcfg.start, task_name)
                if not data[0]:
                    logging.debug("\tlast probability=", data[-1])
                    logging.debug("\tsolution probability=", prob)
                assert data[0] is not None or algorithm != heap_search or prob < data[-1]
            except KeyError as e:
                print("Failed to compute probability of:", task_name)
                print("Error:", e)
        assert algorithm != heap_search or data[-2] <= 1 + 1e-3, data
        successes += data[0] is not None
        output.append((task_name, data))
        pbar.update(1)
        pbar.set_postfix_str(f"{successes} solved")
    pbar.close()
    return output