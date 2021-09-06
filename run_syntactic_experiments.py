import logging
import time
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

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
			result.append(row)

		# if nb_programs % 10_000 == 0:
		# 	logging.debug('tested {} programs'.format(nb_programs))

	return result

timeout = .5
compute_from_scratch = True

for algo_index in range(len(list_algorithms)):
	algorithm, name_algo, param = list_algorithms[algo_index]

	if compute_from_scratch:
		logging.info('start run: {}'.format(name_algo))

		result = []
		if algorithm in randomised:
			for i in range(3):
				result = result + run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
			result = sorted(result, key=lambda x: x[2])
		else:
			result = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)

		logging.info('finished run')

		with open('results_syntactic/{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			header = ['search time', 'probability of the program', 'cumulative probability']
			writer.writerow(header)
			writer.writerows(result)

	else:
		with open('results_syntactic/{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
			reader = csv.reader(f)
			result = []
			for row in reader:
				result.append(row)

	x = [search_time for search_time, probability, cumulative_probability in result]
	y = [cumulative_probability for search_time, probability, cumulative_probability in result]
	plt.scatter(x,y,label = name_algo, s = 5)

#plt.xlim((0,threshold_probability))
plt.legend()
plt.xlabel("time (in seconds)")
plt.ylabel("cumulative probability")
plt.title("")
plt.xscale('log')
# plt.yscale('log')
#plt.show()
plt.savefig("results_syntactic/cumulative_time_%s.png" % seed, 
	dpi=500, 
	bbox_inches='tight')
plt.clf()
