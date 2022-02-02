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
np.random.seed(seed)
deepcoder = DSL(semantics, primitive_types)
type_request = Arrow(List(INT),List(INT))
deepcoder_CFG = deepcoder.DSL_to_CFG(type_request, max_program_depth = 4)
deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

# these colors come from a graphical design webpage
# but I think that they actually look worse
# they are disabled here
six_colors = [None]*6#["#003f5c","#444e86","#955196","#dd5182","#ff6e54","#ffa600"]
seven_colors = [None]*7#["#003f5c","#374c80","#7a5195","#bc5090","#ef5675","#ff764a","#ffa600"]

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
	program = next(gen)
	if algorithm in reconstruct:
		target_type = pcfg.start[0]
		program = reconstruct_from_compressed(program, target_type)
	probability = pcfg.probability_program(pcfg.start, program)
	nb_programs = 1
	cumulative_probability = probability
	seen = set()
	seen.add(program.hash)

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
	number_algorithms = len(list_algorithms)

	timepoints = np.logspace(start = -3, stop = log10(timeout), num = number_timepoints)
	countpoints = np.linspace(start = 0, stop = max_number_programs, num = number_countpoints)

	r_time = np.zeros((number_samples, number_algorithms, number_timepoints))
	r_program = np.zeros((number_samples, number_algorithms, number_countpoints))
	for i in range(number_samples):
		deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

		for algo_index in range(number_algorithms):
			algorithm, name_algo, param = list_algorithms[algo_index]

			logging.info('start run number {}: {}'.format(i+1, name_algo))

			res = run_algorithm(pcfg = deepcoder_PCFG, algo_index = algo_index)
			r_time[i][algo_index] = np.interp(timepoints, 
				[search_time for search_time,_,_ in res], 
				[cumulative_probability for _,_,cumulative_probability in res])			
			r_program[i][algo_index] = np.interp(countpoints, 
				range(len(res)), 
				[cumulative_probability for _,_,cumulative_probability in res])

			logging.info('finished run number {}'.format(i+1))

	result_time_mean = np.mean(r_time, axis=0)
	result_time_std = np.std(r_time, axis=0) 

	result_program_mean = np.mean(r_program, axis=0)
	result_program_std = np.std(r_program, axis=0) 

	for algo_index in range(number_algorithms):
		algorithm, name_algo, param = list_algorithms[algo_index]

		with open('results_syntactic/cumulative_probability_vs_time_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			header = ['search time', 'mean cumulative probability', 'standard deviation']
			writer.writerow(header)
			for x,t in enumerate(timepoints):
				writer.writerow((t, result_time_mean[algo_index][x], result_time_std[algo_index][x]))

		with open('results_syntactic/cumulative_probability_vs_number_programs_{}_{}.csv'.format(name_algo, timeout), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			header = ['number of programs', 'mean cumulative probability', 'standard deviation']
			writer.writerow(header)
			for x in range(number_countpoints):
				writer.writerow((x, result_program_mean[algo_index][x], result_program_std[algo_index][x]))

# Plot cumulative probability VS time
def plot_cumulative_probability_vs_time():
	logging.info('Plot cumulative probability VS time')
	plt.style.use('seaborn-colorblind')

	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		# timepoints = np.arange(start = 0, stop = number_timepoints)
		timepoints = np.logspace(start = -3, stop = log10(timeout), num = number_timepoints)

		logging.info('retrieve run: {}'.format(name_algo))

		with open('results_syntactic/cumulative_probability_vs_time_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
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
		sc = plt.scatter(timepoints, result_mean, label = name_algo, s = 5)
		color = sc.get_facecolors()[0].tolist()
		plt.fill_between(timepoints, result_top, result_low, facecolor = color, alpha=0.2)

	plt.legend()
	plt.xlim((1e-3,timeout))
	plt.xlabel('time (in seconds)')
	plt.xscale('log')
	plt.ylim((0,1))
	plt.ylabel('cumulative probability')
	plt.grid()

	plt.savefig("results_syntactic/cumulative_probability_vs_time_%s.png" % seed,
		dpi=500, 
		bbox_inches='tight')
	plt.clf()


# Plot cumulative probability VS number of programs
def plot_cumulative_probability_vs_number_programs():
	logging.info('Plot cumulative probability VS number of programs')
	plt.style.use('seaborn-colorblind')
	countpoints = np.linspace(start = 0, stop = max_number_programs, num = number_countpoints)

	for algo_index in range(len(list_algorithms)):
		algorithm, name_algo, param = list_algorithms[algo_index]
		# heap search and A* are the same here
		if name_algo == 'A*':
			continue

		logging.info('retrieve run: {}'.format(name_algo))

		with open('results_syntactic/cumulative_probability_vs_number_programs_{}_{}.csv'.format(name_algo, timeout), 'r', encoding='UTF8', newline='') as f:
			reader = csv.reader(f)
			result_mean = np.zeros(number_countpoints)
			result_std = np.zeros(number_countpoints)
			for i, row in enumerate(reader):
				if i == 0:
					continue
				result_mean[i-1] = row[1]
				result_std[i-1] = row[2]

		logging.info('retrieved')
	
		result_top = result_mean + .5 * result_std
		result_low = result_mean - .5 * result_std
		sc = plt.scatter(countpoints,result_mean,label = name_algo, s = 5)
		color = sc.get_facecolors()[0].tolist()
		plt.fill_between(countpoints, result_top, result_low, facecolor = color, alpha=0.2)

	plt.ticklabel_format(axis='x', style='sci', scilimits=(3,5))
	plt.xlabel('number of programs')
	plt.xlim((0,max_number_programs))
	plt.ylabel('cumulative probability')
	plt.ylim((0,1))
	plt.legend(loc = 'lower right')
	plt.grid()

	plt.savefig("results_syntactic/cumulative_probability_vs_number_programs_%s.png" % seed, 
		dpi=500, 
		bbox_inches='tight')
	plt.clf()

number_samples = 50

number_timepoints = 1_000
timeout = 1

number_countpoints = 1_000
max_number_programs = 2e5

create_dataset()
plot_cumulative_probability_vs_time()
plot_cumulative_probability_vs_number_programs()
