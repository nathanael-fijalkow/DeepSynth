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
	(bfs, 'BFS', {'beam_width' : 5e5}),
	(dfs, 'DFS', {}),
	(sort_and_add, 'Sort&Add', {}),
	# (sqrt_sampling_with_sbsur, 'SQRT+SBS', {}),
	(threshold_search, 'Threshold', {'initial_threshold' : 1e-4, 'scale_factor' : 5e3}),
	(sqrt_sampling, 'SQRT', {}),
	(heap_search, 'Heap Search', {}),
	(a_star, 'A*', {}),
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

	# To remove the cost of initialisation
	_ = next(gen)

	nb_programs = 1
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


def create_dataset():
	logging.info('Create dataset')
	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		timepoints = np.linspace(start = 0, stop = timeout, num = number_timepoints)
		countpoints = np.linspace(start = 0, stop = max_number_programs, num = number_countpoints)

		logging.info('start run: {}'.format(name_algo))

		if algorithm in randomised:
			r_time = np.zeros((number_samples, number_timepoints))
			r_program = np.zeros((number_samples, number_countpoints))
			for i in range(number_samples):
				res = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
				r_time[i] = np.interp(timepoints, 
					[search_time for search_time,_,_ in res], 
					[cumulative_probability for _,_,cumulative_probability in res])			
				r_program[i] = np.interp(countpoints, 
					range(len(res)), 
					[cumulative_probability for _,_,cumulative_probability in res])
			logging.info('finished run')

			result_time_mean = np.mean(r_time, axis=0)
			result_time_std = np.std(r_time, axis=0) 
			result_time = np.column_stack((timepoints, result_time_mean, result_time_std))

			result_program_mean = np.mean(r_program, axis=0)
			result_program_std = np.std(r_program, axis=0) 
			result_program = np.column_stack((countpoints, result_program_mean, result_program_std))

			with open('results_syntactic/cumulative_probability_vs_time_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
				writer = csv.writer(f)
				header = ['search time', 'mean cumulative probability', 'standard deviation']
				writer.writerow(header)
				writer.writerows(result_time)

			with open('results_syntactic/cumulative_probability_vs_number_programs_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
				writer = csv.writer(f)
				header = ['number of programs', 'mean cumulative probability', 'standard deviation']
				writer.writerow(header)
				writer.writerows(result_program)

		else:
			result = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
			logging.info('finished run')
			with open('results_syntactic/cumulative_probability_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
				writer = csv.writer(f)
				header = ['search time', 'probability', 'cumulative probability']
				writer.writerow(header)
				writer.writerows(result)

# Plot cumulative probability VS time
def plot_cumulative_probability_vs_time():
	logging.info('Plot cumulative probability VS time')
	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		timepoints = np.linspace(start = 0, stop = timeout, num = number_timepoints)

		logging.info('retrieve run: {}'.format(name_algo))

		if algorithm in randomised:
			with open('results_syntactic/cumulative_probability_vs_time_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
				reader = csv.reader(f)
				result_mean = np.zeros(number_timepoints)
				result_std = np.zeros(number_timepoints)
				for i, row in enumerate(reader):
					if i == 0:
						continue
					result_mean[i-1] = row[1]
					result_std[i-1] = row[2]
		else:
			with open('results_syntactic/cumulative_probability_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
				reader = csv.reader(f)
				result = []
				for i, row in enumerate(reader):
					if i == 0:
						continue
					result.append(row)

		logging.info('retrieved')
	
		if algorithm in randomised:
			result_top = result_mean + 2 * result_std
			result_low = result_mean - 2 * result_std
			plt.fill_between(timepoints, result_top, result_low, alpha=0.2)
			sc = plt.scatter(timepoints,result_mean,label = name_algo, s = 5)
			color = sc.get_facecolors()[0].tolist()
			plt.fill_between(timepoints, result_top, result_low, facecolor = color, alpha=0.2)
		else:
			x = [float(search_time) for search_time, _, _ in result]
			y = [float(cumulative_probability) for _, _, cumulative_probability in result]
			plt.scatter(x,y,label = name_algo, s = 5)

	plt.legend()
	plt.xlim((1e-4,timeout))
	plt.xlabel('time (in seconds)')
	plt.ylabel('cumulative probability')
	plt.xscale('log')
	# plt.yscale('log')

	plt.savefig("results_syntactic/cumulative_probability_vs_time_%s.png" % seed, 
		dpi=500, 
		bbox_inches='tight')
	plt.clf()



# Plot cumulative probability VS number of programs
def plot_cumulative_probability_vs_number_programs():
	logging.info('Plot cumulative probability VS number of programs')
	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		# heap search and A* are the same here
		if name_algo == 'A*':
			continue

		countpoints = np.linspace(start = 0, stop = max_number_programs, num = number_countpoints)

		logging.info('retrieve run: {}'.format(name_algo))

		if algorithm in randomised:
			with open('results_syntactic/cumulative_probability_vs_number_programs_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
				reader = csv.reader(f)
				result_mean = np.zeros(number_countpoints)
				result_std = np.zeros(number_countpoints)
				for i, row in enumerate(reader):
					if i == 0:
						continue
					result_mean[i-1] = row[1]
					result_std[i-1] = row[2]
		else:
			with open('results_syntactic/cumulative_probability_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
				reader = csv.reader(f)
				result = []
				for i, row in enumerate(reader):
					if i == 0:
						continue
					result.append(row)

		logging.info('retrieved')
	
		if algorithm in randomised:
			result_top = result_mean + 2 * result_std
			result_low = result_mean - 2 * result_std
			print(result_top[:10])
			print(result_low[:10])
			sc = plt.scatter(countpoints,result_mean,label = name_algo, s = 5)
			color = sc.get_facecolors()[0].tolist()
			plt.fill_between(countpoints, result_top, result_low, facecolor = color, alpha=0.2)
		else:
			x = np.arange(len(result))
			y = [float(cumulative_probability) for _, _, cumulative_probability in result]
			plt.scatter(x,y,label = name_algo, s = 5)

	plt.ticklabel_format(axis='x', style='sci', scilimits=(3,5))
	plt.xlabel('number of programs')
	plt.xlim((0,max_number_programs))
	plt.ylabel('cumulative probability')
	plt.legend(loc = 'lower right')

	plt.savefig("results_syntactic/cumulative_probability_vs_number_programs_%s.png" % seed, 
		dpi=500, 
		bbox_inches='tight')
	plt.clf()

number_samples = 5

number_timepoints = 100_000
timeout = 10

number_countpoints = 10_000
max_number_programs = 2e5

# create_dataset()
plot_cumulative_probability_vs_time()
plot_cumulative_probability_vs_number_programs()