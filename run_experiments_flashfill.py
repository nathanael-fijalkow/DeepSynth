import torch
from experiment_helper import task_set2dataset
from model_loader import build_flashfill_generic_model, get_model_name
import csv
import os

from run_experiment import gather_data, list_algorithms
from flashfill_dataset_loader import  load_tasks

datataset_name = "flashfill"
save_folder = "."

tasks = load_tasks()
print("Loaded", len(tasks), "tasks")

dsl, cfg, model = build_flashfill_generic_model()
# tasks = filter_tasks_for_model(tasks, model)
print("Remaining tasks after filter:", len(tasks), "tasks")
dataset = task_set2dataset(tasks, model, dsl)
model_name = get_model_name(model)

# Data gathering

for algo_index in range(len(list_algorithms)):
    algo_name = list_algorithms[algo_index][1]
    filename = f"{save_folder}/algo_{algo_name}_model_{model_name}_dataset_{datataset_name}_results_semantic.csv"
    if os.path.exists(filename):
        continue
    data = gather_data(dataset, algo_index)
    rows = [[el[0]] + list(el[1]) for el in data]
    with open(filename, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["task_name", "program", "search_time", "evaluation_time",
                        "nb_programs", "cumulative_probability", "probability"])
        writer.writerows(rows)
