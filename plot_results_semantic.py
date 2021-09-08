import glob
import csv

import matplotlib.pyplot as plt

dataset = "dreamcoder"
model = "fixed+rnn+global"
folder = "results_semantics"


data = {}

# Load data
for file in glob.glob(f"{folder}/*.csv"):
    file_name = file.replace(folder + "/", "")
    if not file_name.startswith("algo_") or not file_name.endswith(f"_model_{model}_dataset_{dataset}_results_semantic.csv"):
        continue
    algo_name = file_name[5:file_name.find("_model")]
    with open(file) as fd:
        reader = csv.reader(fd)
        
        algo_data = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            algo_data.append(row)
        data[algo_name] = algo_data

print("Found data for:", list(data.keys()))

total_tasks = None
# Preprocess
processed_data = {}
for algo_name, algo_data in data.items():
    if total_tasks is None:
        total_tasks = len(algo_data)
    new_data = [[0, 0, 0]]
    cur_succ, curr_time, curr_programs = 0, 0, 0
    for _, prog, search_time, evaluation_time, nb_programs, cumulative_probability, probability in algo_data:
        cur_succ += prog is not None and len(prog) > 0
        curr_time +=  float(search_time) + float(evaluation_time)
        curr_programs += int(nb_programs)
        new_data.append([cur_succ, curr_time, curr_programs])
    processed_data[algo_name] = new_data


plt.style.use('seaborn-colorblind')

# Plot success wrt time
time_max = 0
for algo, data in processed_data.items():
    time_data = [x[1] for x in data]
    time_max = max(time_max, max(time_data))
    plt.plot(time_data, [x[0] for x in data], label=algo)

plt.hlines([total_tasks], xmin=0, xmax=time_max, label="All tasks",
           color=f"C{len(processed_data)}", linestyles="dashed")
plt.xlabel("time (in seconds)")
plt.ylabel("tasks completed")

plt.legend()
plt.show()
