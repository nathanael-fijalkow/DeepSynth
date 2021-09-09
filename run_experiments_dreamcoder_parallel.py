import torch
import csv
import os

from run_experiment import gather_data_parallel, list_algorithms
from model_loader import build_dreamcoder_intlist_model, get_model_name
from experiment_helper import task_set2dataset
from dreamcoder_dataset_loader import load_tasks, filter_tasks_for_model

datataset_name = "dreamcoder"
save_folder = "."

dsl, cfg, model = build_dreamcoder_intlist_model()
tasks = load_tasks()
print("Loaded", len(tasks), "tasks")
tasks = filter_tasks_for_model(tasks, model)
print("Remaining tasks after filter:", len(tasks), "tasks")

dataset = task_set2dataset(tasks, model, dsl)

model_name = get_model_name(model)



# Data gathering
for algo_index in range(len(list_algorithms)):
    algo_name = list_algorithms[algo_index][1]
    if algo_name != "heap search":
        continue
    for splits in [2]:
        filename = f"{save_folder}/algo_{algo_name} {splits} CPUs_model_{model_name}_dataset_{datataset_name}_results_semantic.csv"
        if os.path.exists(filename):
            continue
        data = gather_data_parallel(dataset, algo_index, splits, n_filters=3)

        output = [[el[0]] + list(el[1]) for el in data]
        col_names = ["task_name", "program", "grammar_split_time", "search_time", "evaluation_time",
                    "nb_programs", "cumulative_probability", "probability"]
        processed_data = []
        for row in output:
            current = row[:3]
            search_times = row[3]
            current.append(sum(search_times) / len(search_times))
            evaluation_times = row[4]
            current.append(sum(evaluation_times) / len(evaluation_times))
            nb_programs = row[5]
            current.append(sum(nb_programs))
            probs = row[6]
            current.append(sum(probs))
            current.append(row[7:])

            for li in [search_times, evaluation_times, nb_programs, probs]:
                for el in li:
                    current.append(el)

            processed_data.append(current)

        n_producers = len(output[0][3])
        n_filters = len(output[0][4])
        for name, cnt in [("search_time", n_producers), ("evaluation_time", n_filters), ("nb_programs", n_producers), ("cumulative_probability", n_producers)]:
            for i in range(cnt):
                col_names.append(name + f"_{i}")

        with open(filename, "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(col_names)
            writer.writerows(processed_data)
