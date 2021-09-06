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
from Algorithms.heap_search_naive import heap_search_naive
from Algorithms.a_star import a_star
from Algorithms.threshold_search import threshold_search
from Algorithms.dfs import dfs
from Algorithms.bfs import bfs
from Algorithms.sort_and_add import sort_and_add
from Algorithms.sqrt_sampling import sqrt_sampling, sqrt_sampling_with_sbsur

from DSL.deepcoder import semantics,primitive_types

logging_levels = {0: logging.INFO, 1: logging.DEBUG}
verbosity = 0
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

seed = 100
random.seed(seed)
deepcoder = DSL(semantics, primitive_types)
type_request = Arrow(List(INT),List(INT))
deepcoder_CFG = deepcoder.DSL_to_CFG(type_request, max_program_depth = 4)
deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

list_algorithms = [
	(heap_search, 'Heap Search', {}),
	(sqrt_sampling, 'SQRT', {}),
	# (sqrt_sampling_with_sbsur, 'SQRT+SBS', {}),
	# (a_star, 'A*', {}),
	# (threshold_search, 'Threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}),
	# (bfs, 'BFS', {'beam_width' : 500_000}),
	# (dfs, 'DFS', {}),
	# (sort_and_add, 'Sort&Add', {}),
]
# Set of algorithms where we need to reconstruct the programs
reconstruct = {dfs, bfs, threshold_search, a_star,
			   sort_and_add, sqrt_sampling_with_sbsur}
# Set of randomised algorithms
randomised = {sqrt_sampling, sqrt_sampling_with_sbsur}

def run_algorithm(pcfg, algo_index):
	'''
	Run the algorithm until timeout, and for each program record probability and time of output
	'''
	algorithm, name_algo, param = list_algorithms[algo_index]
	result = []

	search_time = 0
	gen = algorithm(pcfg, **param)
	found = False
	if name_algo == "SQRT":
		_ = next(gen)

	nb_programs = 0
	cumulative_probability = 0
	seen = set()

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

		# Reconstruction if needed
		if algorithm in reconstruct:
			target_type = pcfg.start[0]
			program = reconstruct_from_compressed(program, target_type)

		if program.hash not in seen:
			seen.add(program.hash)
			probability = pcfg.probability_program(pcfg.start, program)
			cumulative_probability += probability
			nb_programs += 1
			row = search_time, probability, cumulative_probability
			# row = search_time, probability, log10(1 - cumulative_probability)
			result.append(row)

		# if nb_programs % 10_000 == 0:
		# 	logging.debug('tested {} programs'.format(nb_programs))

	return result


def plot():
	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		interval = np.linspace(start = 0, stop = timeout, num = number_points)

		if compute_from_scratch:
			logging.info('start run: {}'.format(name_algo))

			if algorithm in randomised:
				r = np.zeros((number_samples, number_points))
				for i in range(number_samples):
					res = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
					r[i] = np.interp(interval, [x[0] for x in res], [x[2] for x in res])			
				logging.info('finished run')
				result_mean = np.mean(r, axis=0)
				result_std = np.std(r, axis=0) 
				result = np.column_stack((interval, result_mean,result_std))

				with open('results_syntactic/{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
					writer = csv.writer(f)
					header = ['search time', 'mean cumulative probability', 'standard deviation']
					writer.writerow(header)
					writer.writerows(result)

			else:
				res = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
				logging.info('finished run')
				result = [(x[0], x[2]) for x in res]
				with open('results_syntactic/{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
					writer = csv.writer(f)
					header = ['search time', 'cumulative probability']
					writer.writerow(header)
					writer.writerows(result)

		else:
			logging.info('retrieve run: {}'.format(name_algo))
			with open('results_syntactic/{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
				reader = csv.reader(f)
				if algorithm in randomised:
					result_mean = np.zeros(number_points)
					result_std = np.zeros(number_points)
					for i, row in enumerate(reader):
						if i > 0:
							result_mean[i-1] = row[1]
							result_std[i-1] = row[2]
				else:
					result = []
					for row in reader:
						result.append(row)
			logging.info('retrieved')

		if algorithm in randomised:
			result_top = result_mean + 2 * result_std
			result_low = np.positive(result_mean - 2 * result_std)
			plt.fill_between(interval, result_top, result_low, facecolor='orange', alpha=0.2)
			plt.scatter(interval,result_mean,label = name_algo, s = 5)
		else:
			x = [x[0] for x in result]
			y = [x[1] for x in result]
			plt.scatter(x,y,label = name_algo, s = 5)

	plt.legend()
	plt.xlabel("time (in seconds)")
	plt.ylabel("cumulative probability")
	# plt.ylabel("log(1 - cumulative probability)")
	plt.title("")
	plt.xscale('log')
	# plt.yscale('log')
	#plt.show()
	plt.savefig("results_syntactic/cumulative_time_%s.png" % seed, 
		dpi=500, 
		bbox_inches='tight')
	plt.clf()

timeout = 5
compute_from_scratch = True
number_points = 100_000
number_samples = 10
plot()
