import pickle
import time
import random
import matplotlib.pyplot as plt
from math import log10

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from program import Program, Function, Variable, BasicPrimitive, New
from cfg import CFG
from pcfg import PCFG
from dsl import DSL

from DSL.deepcoder import semantics,primitive_types
from Algorithms.heap_search import heap_search
from Algorithms.a_star import a_star
from Algorithms.threshold_search import threshold_search
from Algorithms.dfs import dfs
from Algorithms.bfs import bfs
from Algorithms.sqrt_sampling import sqrt_sampling

seed = 100
random.seed(seed)
deepcoder = DSL(semantics, primitive_types, no_repetitions)
t = Arrow(List(INT),List(INT))
deepcoder_PCFG_t = deepcoder.DSL_to_Random_PCFG(t, alpha = .7)


# first experiment: x = time, y = cumulative proba
# second experiment: x = number of programs enumerated, y = time
# third experiment: x = proba program, y = time/average time to find the program

# Set of algorithms where we need to reconstruct the programs
reconstruct = {dfs, bfs, threshold_search, a_star}

def create_dataset(PCFG):
	'''
	Create a dataset, which is a list of number_samples programs with proba in [1O^(-(i+1),1O^(-i)] for i in [imin, imax]
	'''
	dataset = []
	size_dataset = [0 for _ in range(imax)]
	finished = False

#	gen = PCFG.sampling()
	gen = sqrt_sampling(PCFG) # better than a simple sampling to get interesting programs

	while(not finished):
		program = next(gen) # format: a program
		proba = PCFG.probability_program(PCFG.start, program)
		i = int(-log10(proba))
		if (i >= imin and i < imax and size_dataset[i] < number_samples):
#            print("This program has probability %0.10f, it goes in bucket %u.\n" % (proba, i))
			dataset.append((program,proba))
			size_dataset[i] += 1
			if size_dataset[i] == number_samples:
				j = imin
				finished = True
				while(finished and j < imax):
					finished = size_dataset[j] == number_samples
					j += 1
	# We sort the dataset by decreasing probability
	dataset = sorted(dataset, key = lambda pair: pair[1], reverse = True)
#	print(dataset)
	return dataset


def run_algorithm(dsl, PCFG, algorithm, param):
	'''
	Run the algorithm until either timeout or 3M programs, and for each program record probability and time of output
	'''
	print("Running: %s" % algorithm.__name__)
	# result = []
	# seen = set()
	result = {} # str(prog) : N, chrono, proba
	N = 0
	chrono = 0
	gen = algorithm(PCFG, **param)
	while (chrono < timeout and N < total_number_programs):
		chrono -= time.perf_counter()
		program = next(gen)

		# if algorithm.__name__ == 'dfs':
    	# 		print(program)

		chrono += time.perf_counter()
		if algorithm.__name__ == 'bfs':
    			print(N)
		if algorithm in reconstruct:
			program = dsl.reconstruct_from_compressed(program, PCFG.start[0])
		# if algorithm.__name__ == 'dfs':
    	# 		print(program)
		# if N <= 10:
    	# 		print(algorithm.__name__, program)

		hash_program = str(program)
		if hash_program not in result:
			N += 1
			result[hash_program] = N, chrono, PCFG.probability_program(PCFG.start, program)
			# result.append((program, PCFG.proba_term(PCFG.start, program), chrono))

	print("Run successful, output %u programs" % len(result))
#    print(result)
	return result



def experiment_enumeration_time(result):
	chrono_result = [chrono for (program, probability, chrono) in result]
	return(chrono_result)




####################
# First experiment #
####################
# cumulative probability versus time

# parameters
timeout = 30  # in seconds
random.seed(seed)
total_number_programs = 150_000
dsl = deepcoder
pcfg = deepcoder_PCFG_t
threshold_probability = 1 # do not plot if cumulative proba greater than this threshold, otherwise hard to interpret

#title = "Cumulative probability versus time"
title = ""
recompute_from_scratch = False

#list_algorithms = [(heap_search, 'heap search', {'dsl' : dsl, 'environments': {}}), (dfs, 'dfs', {}), (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}), (sqrt_sampling, 'SQRT', {}), (a_star, 'A*', {})]
list_algorithms = [(heap_search, 'heap search', {}), (sqrt_sampling, 'SQRT', {}), (dfs, 'dfs', {}), (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}), (a_star, 'A*', {}), (bfs, 'bfs', {'beam_width' : 50000})]

if recompute_from_scratch:
	for algo, algo_name, param in list_algorithms:
		run_algo = run_algorithm(dsl, pcfg, algo, param)
		with open("results_syntactic/run_1_" + algo_name + '_' + str(seed) + ".pickle","wb" ) as f:
			pickle.dump(run_algo, f)

def experiment_cumulative_vs_time(run_algo):
	result = [(run_algo[e][0], run_algo[e][1], run_algo[e][2]) for e in run_algo] #N, chrono, proba
	result.sort(key = lambda x: x[0])
	cumulative = 0
	for i in range(0, len(result)):
		proba = result[i][2]
		cumulative+=proba
		result[i] = (cumulative, result[i][1])
		# result.append((cumulative, chrono))
	return result

def plot_cumulative_vs_time(PCFG, list_algorithms):
	'''Retrieve the results and plot'''
	for i, (algorithm, algo_name, param) in enumerate(list_algorithms):
		try:
			f = open("results_syntactic/run_1_" + algo_name + '_' + str(seed) + ".pickle","rb")
			run_algo = pickle.load(f)
			f.close()
		except:
			print("algorithm not run yet")
			assert(False)
		processed_run_algo = experiment_cumulative_vs_time(run_algo)
		plt.scatter([x for (x,y) in processed_run_algo], [y for (x,y) in processed_run_algo], label = algo_name, s = 8)
	#plt.xlim((0,threshold_probability))
	plt.legend()
	plt.xlabel("cumulative probability")
	plt.ylabel("time (in seconds)")
	plt.title(title)
	# plt.xscale('log')
	plt.yscale('log')
	#plt.show()
	plt.savefig("results_syntactic/cumulative_time_%s.png" % seed, dpi=500, bbox_inches='tight')
	plt.clf()

plot_cumulative_vs_time(pcfg, list_algorithms)



# #####################
# # Second experiment #
# #####################
# # Enumeration time: heap search versus A*

# # parameters
# # title = "Heap search versus A*"
# title = ""
# timeout = 50  # in seconds
# random.seed(seed)
# total_number_programs = 300_000
# dsl = deepcoder
# pcfg = deepcoder_PCFG_t
# list_algorithms = [(heap_search, 'heap search', {}), (a_star, 'A*', {})]

# recompute_from_scratch = False

# if recompute_from_scratch:
# 	for algo, algo_name, param in list_algorithms:
# 		run_algo = run_algorithm(dsl, pcfg, algo, param)
# 		with open("results_syntactic/run_2_" + algo_name + '_' + str(seed) + ".pickle","wb" ) as f:
# 			pickle.dump(run_algo, f)

# def experiment_enumeration_time(run_algo):
# 	result = [run_algo[e][1] for e in run_algo] #N, chrono, proba
# 	result.sort()
# 	return result

# def plot_enumeration_time(PCFG, list_algorithms):
# 	'''
# 	Retrieve the results and plot
# 	'''
# 	for i, (algorithm, algo_name, param) in enumerate(list_algorithms):
# 		try:
# 			f = open("results_syntactic/run_2_" + algo_name + '_' + str(seed) + ".pickle","rb")
# 			run_algo = pickle.load(f)
# 			f.close()
# 		except:
# 			print("algorithm not run yet")
# 			assert(False)
# 		processed_run_algo = experiment_enumeration_time(run_algo)
# 		plt.scatter([x for x in range(1,len(processed_run_algo)+1)], processed_run_algo, label = algo_name, s = 8)
# 	#min_proba = 0.5
# 	plt.legend()
# 	plt.xlabel("number of programs")
# 	plt.ylabel("time (in seconds)")
# 	plt.title(title)
# 	#plt.xscale('log')
# 	plt.yscale('log')
# 	#plt.show()
# 	plt.savefig("results_syntactic/enumeration_time_%s.png" % seed, dpi=300, bbox_inches='tight')
# 	plt.clf()

# plot_enumeration_time(pcfg, list_algorithms)




####################
# Third experiment #
####################
# probability programs versus search time
# paramaters for the dataset
#Create a dataset, number_samples programs with proba in [1O^(-(i+1),1O^(-i)] for i in [imin, imax]
imin = 2
imax = 10
number_samples = 10

#title = "probability versus search time"
title = ""
# others parameters
total_number_programs = 150_000
timeout = 20  # in seconds
random.seed(seed)
dsl = deepcoder
pcfg = deepcoder_PCFG_t

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

recompute_from_scratch = True

if recompute_from_scratch:
	dataset = create_dataset(pcfg)
	with open('results_syntactic/dataset_3_'+ str(seed) + ".pickle","wb" ) as f:
		pickle.dump(dataset, f)

	for algo, algo_name, param in list_algorithms:
		run_algo = run_algorithm(dsl, pcfg, algo, param)
		with open("results_syntactic/run_3_" + algo_name + '_' + str(seed) + ".pickle","wb" ) as f:
			pickle.dump(run_algo, f)

def experiment_probability_vs_time(dataset, run_algo):
	result = []
	for program,proba in dataset:
		hash_program = str(program)
		if hash_program in run_algo:
			result.append((proba,run_algo[hash_program][1]))
		else:
			result.append((proba, timeout))
	return result

def plot_probability_vs_time(PCFG, list_algorithms):
	'''
	Retrieve the results and plot
	'''
	with open('results_syntactic/dataset_3_'+ str(seed) + ".pickle", 'rb') as f:
		dataset = pickle.load(f)

	for i, (algorithm, algo_name, param) in enumerate(list_algorithms):
		try:
			f = open("results_syntactic/run_3_" + algo_name + '_' + str(seed) + ".pickle","rb")
			run_algo = pickle.load(f)
			f.close()
		except:
			print("algorithm not run yet")
			assert(False)
		processed_run_algo = experiment_probability_vs_time(dataset, run_algo)
		plt.scatter([x for (x,y) in processed_run_algo], [y for (x,y) in processed_run_algo], label = algo_name, s = 8)
	#min_proba = 0.5
	#plt.xlim((0,min_proba))
	plt.legend()
	plt.xlabel("probability")
	plt.ylabel("search time (in seconds)")
	plt.title(title)
	plt.xscale('log')
	plt.yscale('log')
	#plt.show()
	plt.savefig("results_syntactic/proba_vs_search_time_%s.png" % seed, dpi=300, bbox_inches='tight')
	plt.clf()

plot_probability_vs_time(pcfg, list_algorithms)



#####################
# Fourth experiment #
#####################
# cumulative proba versus number of programs output

# parameters
#title = "cumulative probability versus number of programs output"
title = ""
timeout = 30  # in seconds
random.seed(seed)
total_number_programs = 150_000
dsl = deepcoder
pcfg = deepcoder_PCFG_t
threshold_probability = 1
recompute_from_scratch = False

#list_algorithms = [(heap_search, 'heap search', {'dsl' : dsl, 'environments': {}}), (dfs, 'dfs', {}), (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}), (sqrt_sampling, 'SQRT', {}), (a_star, 'A*', {})]
#list_algorithms = [(heap_search, 'heap search', {}), (sqrt_sampling, 'SQRT', {}), (dfs, 'dfs', {}), (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10})]
list_algorithms = [(heap_search, 'heap search', {}), (sqrt_sampling, 'SQRT', {}), (dfs, 'dfs', {}), (threshold_search, 'threshold', {'initial_threshold' : 0.0001, 'scale_factor' : 10}), (bfs, 'bfs', {'beam_width' : 50000})]

if recompute_from_scratch:
	for algo, algo_name, param in list_algorithms:
		run_algo = run_algorithm(dsl, pcfg, algo, param)
		with open("results_syntactic/run_4_" + algo_name + '_' + str(seed) + ".pickle","wb" ) as f:
			pickle.dump(run_algo, f)

def experiment_cumulative_vs_trials(run_algo):
	result = [(run_algo[e][0], run_algo[e][1], run_algo[e][2]) for e in run_algo] #N, chrono, proba
	result.sort(key = lambda x: x[0])
	cumulative = 0
	for i in range(0, len(result)):
		proba = result[i][2]
		cumulative+=proba
		result[i] = (cumulative, result[i][1])
		# result.append((cumulative, chrono))
	return result

def plot_cumulative_vs_trials(PCFG, list_algorithms):
	'''Retrieve the results and plot'''
	for i, (algorithm, algo_name, param) in enumerate(list_algorithms):
		try:
			f = open("results_syntactic/run_1_" + algo_name + '_' + str(seed) + ".pickle","rb")
			run_algo = pickle.load(f)
			f.close()
		except:
			print("algorithm not run yet")
			assert(False)
		processed_run_algo = experiment_cumulative_vs_trials(run_algo)
		plt.scatter([x for (x,y) in processed_run_algo], list(range(1,len(processed_run_algo)+1)), label = algo_name, s = 8)
	#plt.xlim((0,threshold_probability))
	plt.legend()
	plt.xlabel("cumulative probability")
	plt.ylabel("number of programs output")
	plt.title(title)
	# plt.xscale('log')
	plt.yscale('log')
	#plt.show()
	plt.savefig("results_syntactic/cumulative_time_versus_trials_%s.png" % seed, dpi=500, bbox_inches='tight')
	plt.clf()

plot_cumulative_vs_trials(pcfg, list_algorithms)


