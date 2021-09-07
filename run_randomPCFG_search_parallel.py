import torch
from ray.util.queue import Empty
from Algorithms.ray_parallel import make_parallel_pipelines, start
from cons_list import cons_list2list
import logging
import time
import random
import csv
import typing
import matplotlib.pyplot as plt
import numpy as np
from math import log10

import ray

import grammar_splitter

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

from DSL.deepcoder import semantics, primitive_types

logging_levels = {0: logging.INFO, 1: logging.DEBUG}
verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])
split_numbers = [2, 6]
seed = 100
random.seed(seed)
np.random.seed(seed)
deepcoder = DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
deepcoder_CFG = deepcoder.DSL_to_CFG(type_request, max_program_depth=4)
deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

# these colors come from a graphical design webpage
# but I think that they actually look worse
# they are disabled here
# ["#003f5c","#444e86","#955196","#dd5182","#ff6e54","#ffa600"]
six_colors = [None]*6
# ["#003f5c","#374c80","#7a5195","#bc5090","#ef5675","#ff764a","#ffa600"]
seven_colors = [None]*7

list_algorithms = [
    # (bfs, 'BFS', {'beam_width' : 5e5}),
    # (dfs, 'DFS', {}),
    # (sort_and_add, 'Sort&Add', {}),
    # (sqrt_sampling_with_sbsur, 'SQRT+SBS', {}),
    # (threshold_search, 'Threshold', {'initial_threshold' : 1e-4, 'scale_factor' : 5e3}),
    (sqrt_sampling, 'SQRT', {}),
    (heap_search, 'Heap Search', {}),
    # (a_star, 'A*', {}),
]
# Set of algorithms where we need to reconstruct the programs
reconstruct = {dfs, bfs, threshold_search, a_star,
               sort_and_add, sqrt_sampling_with_sbsur}
# Set of randomised algorithms
randomised = {sqrt_sampling, sqrt_sampling_with_sbsur}


def insert_prefix(prefix, prog):
    try:
        head, tail = prog
        return (head, insert_prefix(prefix, tail))
    except:
        return prefix


def reconstruct_from_list(program_as_list, target_type):
    if len(program_as_list) == 1:
        return program_as_list.pop()
    else:
        P = program_as_list.pop()
        if isinstance(P, (New, BasicPrimitive)):
            list_arguments = P.type.ends_with(target_type)
            arguments = [None] * len(list_arguments)
            for i in range(len(list_arguments)):
                arguments[len(list_arguments) - i - 1] = reconstruct_from_list(
                    program_as_list, list_arguments[len(
                        list_arguments) - i - 1]
                )
            return Function(P, arguments)
        if isinstance(P, Variable):
            return P

        assert False, f"List: {program_as_list} P:{P} ({type(P)}"


def insert_prefix_toprog(prefix, prog, target_type):
    prefix = cons_list2list(prefix)
    return reconstruct_from_list([prog] + prefix, target_type)


def run_algorithm_parallel(pcfg: PCFG, algo_index: int, splits: int,
                           n_filters: int = 4, transfer_queue_size: int = 500_000, transfer_batch_size: int = 10) -> typing.Tuple[Program, float, typing.List[float], typing.List[float], typing.List[int], typing.List[float], float]:
    '''
    Run the algorithm until either timeout or 1M programs, and for each program record probability and time of output
    return program, search_time, evaluation_time, nb_programs, cumulative_probability, probability
    '''
    algorithm, _, param = list_algorithms[algo_index]

    @ray.remote
    class DataCollectorActor:
        def __init__(self, n):
            self.search_times = [0] * n
            self.times = []
            self.programs = 0
            self.prob = 0

        def add_search_data(self, i, t, probability) -> bool:
            self.search_times[i] += t
            mini = np.mean(self.search_times)
            self.times.append(mini)
            if probability > 0:
                self.programs += 1
            # self.probabilities.append(self.prob)
            if self.search_times[i] > timeout or self.programs > max_number_programs:
                # print("Should stop because: time=", self.total_time,
                #       "programs=", self.programs, "prob=", self.prob)
                return True
            return False

        def search_data(self):
            return self.times

    data_collector = DataCollectorActor.remote(splits)

    def bounded_generator(prefix, cur_pcfg, i):
        if algorithm in reconstruct:
            def new_gen():
                gen = algorithm(cur_pcfg, **param)
                p = next(gen)
                data_collector.add_search_data.remote(i, 0, 1)
                try:
                    while True:
                        t = -time.perf_counter()
                        p = next(gen)
                        prog = insert_prefix(prefix, p)
                        t += time.perf_counter()
                        # probability = pcfg.probability_program(pcfg.start, prog_r)
                        if ray.get(data_collector.add_search_data.remote(i, t, 1)):
                            break
                        # yield prog
                except StopIteration:
                    return
                yield None
        else:
            def new_gen():
                gen = algorithm(cur_pcfg)
                target_type = pcfg.start[0]
                p = next(gen)
                data_collector.add_search_data.remote(i, 0, 1)
                try:
                    while True:
                        t = -time.perf_counter()
                        p = next(gen)
                        if p is None:
                            t += time.perf_counter()
                            if ray.get(data_collector.add_search_data.remote(i, t, 0)):
                                break
                            continue
                        if prefix is None:
                            prog = p
                            t += time.perf_counter()
                        else:
                            prog = insert_prefix_toprog(
                                prefix, p, target_type)
                            t += time.perf_counter()
                        # probability = pcfg.probability_program(
                        #     pcfg.start, prog)

                        if ray.get(data_collector.add_search_data.remote(i, t, 1)):
                            break
                        # yield prog
                except StopIteration:
                    pass
                yield None
        return new_gen

    grammar_split_time = - time.perf_counter()
    splits = grammar_splitter.split(pcfg, splits, alpha=1.05)
    grammar_split_time += time.perf_counter()
    make_generators = [bounded_generator(
        prefix, pcfg, i) for i, (prefix, pcfg) in enumerate(splits)]

    def make_filter(i):
        return lambda x: False

    producers, filters, transfer_queue, out = make_parallel_pipelines(
        make_generators, make_filter, n_filters, transfer_queue_size, splits, transfer_batch_size, filters_stop_number=200)
    start(filters)
    logging.debug("\tStarted {} filters.".format(len(filters)))
    start(producers)
    logging.debug("\tStarted {} producers.".format(len(producers)))

    while True:
        try:
            out.get(timeout=5)
        except Empty:
            pass
        times = ray.get(data_collector.search_data.remote())
        print("Current state: time used:", times[-1], "programs:", len(times))
        if times[-1] > timeout or len(times) > max_number_programs:
            break

    search_times = ray.get(
        data_collector.search_data.remote())

    # Shutdown
    for producer in producers:
        try:
            ray.kill(producer)
        except ray.exceptions.RayActorError:
            continue
    for filter in filters:
        try:
            ray.kill(filter)
        except ray.exceptions.RayActorError:
            continue
    transfer_queue.shutdown(True)
    out.shutdown(True)

    logging.info("\tShut down.")

    return search_times


def create_dataset():
    logging.info('Create dataset')
    number_algorithms = len(list_algorithms)

    timepoints = np.logspace(
        start=-2, stop=log10(timeout), num=number_timepoints)
    r_program = np.zeros(
        (number_samples, len(split_numbers) * number_algorithms, number_countpoints))
    for i in range(number_samples):
        deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

        for j, splits in enumerate(split_numbers):
            for algo_index in range(number_algorithms):
                algorithm, name_algo, param = list_algorithms[algo_index]

                logging.info('start run number {}: {} {}'.format(
                    i+1, name_algo, splits))

                res = run_algorithm_parallel(
                    pcfg=deepcoder_PCFG, algo_index=algo_index, splits=splits, n_filters=1)
                # r_time[i][algo_index] = np.interp(timepoints,
                #                                   [search_time for search_time,
                #                                    _ in res],
                #                                   [cumulative_probability for _, cumulative_probability in res])
                r_program[i][algo_index * number_algorithms + j] = np.interp(timepoints,
                                                                             res,
                                                                            range(len(res)))

                logging.info('finished run number {}'.format(i+1))

        # result_time_mean = np.mean(r_time, axis=0)
        # result_time_std = np.std(r_time, axis=0)

    result_program_mean = np.mean(r_program, axis=0)
    result_program_std = np.std(r_program, axis=0)

    for algo_index in range(number_algorithms):
        algorithm, name_algo, param = list_algorithms[algo_index]
        for j, splits in enumerate(split_numbers):
            index = algo_index * number_algorithms + j
            with open('results_syntactic/time_vs_number_programs_{}_{}_splits{}.csv'.format(name_algo, timeout, splits), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                header = ['time', 'mean number of programs'
                          , 'standard deviation']
                writer.writerow(header)
                for x, t in enumerate(timepoints):
                    writer.writerow(
                        (t, result_program_mean[index][x], result_program_std[index][x]))


def plot_programs_vs_time():
    logging.info('Plot programs VS time')
    for splits in split_numbers:
        for algo_index in range(len(list_algorithms)):
            algorithm, name_algo, param = list_algorithms[algo_index]
            # timepoints = np.arange(start = 0, stop = number_timepoints)
            timepoints = np.logspace(
                start=-2, stop=log10(timeout), num=number_timepoints)

            logging.info('retrieve run: {} {}'.format(name_algo, splits))

            with open('results_syntactic/time_vs_number_programs_{}_{}_splits{}.csv'.format(name_algo, timeout, splits), 'r', encoding='UTF8', newline='') as f:
                reader = csv.reader(f)
                result_mean = np.zeros(number_timepoints)
                result_std = np.zeros(number_timepoints)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    result_mean[i-1] = row[1]
                    result_std[i-1] = row[2]

            logging.info('retrieved')

            result_top = result_mean + .5 * result_std
            result_low = result_mean - .5 * result_std
            name_algo += f" {splits} CPUs"
            sc = plt.scatter(timepoints, result_mean, label=name_algo, s=5)
            color = sc.get_facecolors()[0].tolist()
            plt.fill_between(timepoints, result_top, result_low,
                             facecolor=color, alpha=0.2)

    plt.legend()
    plt.xlim((1e-2, timeout))
    plt.xlabel('time (in seconds)')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim((0, max_number_programs))
    plt.ylabel('number of programs')

    plt.savefig("results_syntactic/programs_vs_time_%s_parallel.png" % seed,
                dpi=500,
                bbox_inches='tight')
    plt.clf()


number_samples = 5

number_timepoints = 1_000
timeout = 1

number_countpoints = 1_000
max_number_programs = 1e7
ray.init()
create_dataset()
plot_programs_vs_time()
