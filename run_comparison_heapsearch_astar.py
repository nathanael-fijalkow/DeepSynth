import logging
import time
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from math import log10

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from program import Program, Function, Variable, BasicPrimitive, New
from cfg import CFG
from pcfg import PCFG
from dsl import DSL
from program_as_list import reconstruct_from_compressed

from Algorithms.heap_search import heap_search
# from Algorithms.heap_search_naive import heap_search_naive
from Algorithms.a_star import a_star

from DSL.deepcoder import semantics,primitive_types

logging_levels = {0: logging.INFO, 1: logging.DEBUG}
verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

seed = 100
random.seed(seed)
deepcoder = DSL(semantics, primitive_types)
type_request = Arrow(List(INT),List(INT))
deepcoder_CFG = deepcoder.DSL_to_CFG(type_request, max_program_depth = 4)

list_algorithms = [
	# (heap_search_naive, 'Heap Search naive', {}),
	(heap_search, 'Heap Search', {}),
	(a_star, 'A*', {}),
]
reconstruct = {a_star}

def run_algorithm(pcfg, algo_index):
	'''
	Run the algorithm until timeout, and for each program record probability and time of output
	'''
	algorithm, name_algo, param = list_algorithms[algo_index]
	result = []

	search_time = 0
	gen = algorithm(pcfg, **param)

	nb_programs = 0
	cumulative_probability = 0

	while (search_time < timeout):
		search_time -= time.perf_counter()
		try:
			program = next(gen)
		except StopIteration:
			search_time += time.perf_counter()
			logging.debug(
				"Output the last program after {}".format(nb_programs))
			break  # no next program
		search_time += time.perf_counter()
		# logging.debug('program found: {}'.format(program))

		if algorithm in reconstruct:
			target_type = pcfg.start[0]
			program = reconstruct_from_compressed(program, target_type)

		probability = pcfg.probability_program(pcfg.start, program)
		cumulative_probability += probability
		nb_programs += 1
		row = search_time, probability, cumulative_probability
		result.append(row)

	return result


def create_dataset():
	logging.info('Create dataset')
	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		timepoints = np.linspace(start = 0, stop = timeout, num = number_timepoints)

		logging.info('start run: {}'.format(name_algo))
		r_program = np.zeros((number_samples, number_timepoints))
		for i in range(number_samples):
			deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()
			res = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
			r_program[i] = np.interp(timepoints, 
				[search_time for search_time, _, _ in res],
				range(len(res)))
		logging.info('finished run')
		result_mean = np.mean(r_program, axis=0)
		result_std = np.std(r_program, axis=0) 
		result = np.column_stack((timepoints, result_mean, result_std))

		with open('results_syntactic/run_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			header = ['time', 'mean number of programs', 'standard deviation']
			writer.writerow(header)
			writer.writerows(result)

# Plot comparison
def plot():
	logging.info('Plot comparison')
	timepoints = np.linspace(start = 0, stop = timeout, num = number_timepoints)

	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]

		logging.info('retrieve run: {}'.format(name_algo))

		with open('results_syntactic/run_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
			reader = csv.reader(f)
			result_mean = np.zeros(number_timepoints)
			result_std = np.zeros(number_timepoints)
			for i, row in enumerate(reader):
				if i == 0:
					continue
				result_mean[i-1] = row[1]
				result_std[i-1] = row[2]

		logging.info('retrieved')
	
		result_top = result_mean + 2 * result_std
		result_low = np.positive(result_mean - 2 * result_std)
		sc = plt.scatter(timepoints, result_mean, label = name_algo, s = 5)
		color = sc.get_facecolors()[0].tolist()
		plt.fill_between(timepoints, result_top, result_low, facecolor = color, alpha=0.2)

	plt.legend(loc = 'top left')
	plt.xlim((1e-1,timeout))
	plt.ticklabel_format(axis='y', style='sci', scilimits=(3,5))
	plt.xlabel('time (in seconds)')
	plt.ylabel('number of programs')
	plt.xscale('log')

	plt.savefig("results_syntactic/comparison_heapsearch_astar%s.png" % seed, 
		dpi=500, 
		bbox_inches='tight')
	plt.clf()

timeout = 10

number_samples = 5

number_timepoints = 10_000
max_number_programs = 1_000_000

create_dataset()
plot()
