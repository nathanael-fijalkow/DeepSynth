import glob
import csv

import matplotlib.pyplot as plt

import numpy as np

# # Dreamcoder
# dataset = "dreamcoder"
# folder = "results_semantics/dreamcoder"
# # HeapSearch Evolution
dataset = "dreamcoder_gen"
folder = "."
# # Deepcpder
# dataset = "deepcoder_T=3_test"
# folder = "./results_semantics/deepcoder"

cutoff_time = 101
plot_max = False
max_variables_per_task = 1

data = {}

# Load data
for file in glob.glob(f"{folder}/*.csv"):
    file_name = file.replace(folder + "/", "")
    file_name = file_name.replace(".csv", "")
    if not file_name.startswith("algo_") or f"_dataset_{dataset}_results_semantic" not in file_name:
        continue
    algo_name = file_name[5:file_name.find("_model")]
    with open(file) as fd:
        reader = csv.reader(fd)
        
        algo_data = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            algo_data.append(row)
        if algo_name not in data:
            data[algo_name] = []
        data[algo_name].append(algo_data)

print("Found data for:", list(data.keys()))

total_tasks = None
# Preprocess
processed_data = {}
for algo_name, algo_data in data.items():
    # Compute total_tasks
    total_tasks = max(total_tasks or 0, max(len(run) for run in algo_data))
    # Remove tasks that have more variables
    for name, _, _, _, _, _, _ in algo_data[0]:
        if f"var{max_variables_per_task}" in name:
            total_tasks -= 1
            continue

    output_matrix = np.zeros((len(algo_data), total_tasks, 6), float)
    for i, run in enumerate(algo_data):
        j: int = 0
        for name, prog, search_time, evaluation_time, nb_programs, cumulative_probability, probability in run:
            if f"var{max_variables_per_task}" in name:
                continue
            success: int = prog is not None and len(prog) > 0
            output_matrix[i, j, 0] = (output_matrix[i, j - 1, 0] if j > 0 else 0) + success
            if float(search_time) + float(evaluation_time) >= cutoff_time + 1:
                output_matrix[i, j, 0] -= success
            capped_search_time = min(
                cutoff_time - float(evaluation_time), float(search_time))
            time_used = capped_search_time + float(evaluation_time)
            output_matrix[i, j, 1] = (
                output_matrix[i, j - 1, 1] if j > 0 else 0) + time_used
            output_matrix[i, j, 2] = (
                output_matrix[i, j - 1, 2] if j > 0 else 0) + int(nb_programs)
            programs_per_sec = int(nb_programs) / time_used
            output_matrix[i, j, 3] = capped_search_time
            output_matrix[i, j, 4] = float(evaluation_time)
            output_matrix[i, j, 4] = programs_per_sec
            j += 1
    processed_data[algo_name] = output_matrix


plt.style.use('seaborn-colorblind')
print(f"Algorithm       Prog/s")

# Plot success wrt time
time_max = 0
for algo, data in processed_data.items():
    time_max = max(time_max, np.max(data[:, :, 1]))

    # Mean plot
    mean_time = np.mean(data[:, :, 1], axis=0)
    mean_success = np.mean(data[:, :, 0], axis=0)
    plt.plot(mean_time, mean_success, label=algo)

    std_time = np.std(data[:, :, 1], axis=0)
    plt.fill_betweenx(mean_success, mean_time - 2 *
                      std_time, mean_time + 2 * std_time, alpha=.4)

    programs_per_sec = np.mean(data[:, :, 4])
    print(f"{algo:<15} {programs_per_sec:.0f} prog/s")
if plot_max:
    plt.hlines([total_tasks], label="All tasks",
            color="k", linestyles="dashed", xmin=0, xmax=time_max)
plt.xlabel("time (in seconds)")
plt.ylabel("tasks completed")
plt.xlim(0, time_max * 1.02)
if plot_max:
    plt.ylim(0, total_tasks + 2)
else:
    plt.ylim(0)
plt.grid()
plt.legend()
plt.savefig(f"results_semantics/machine_learned_{dataset}.png",
		dpi=500, 
		bbox_inches='tight')
plt.show()
