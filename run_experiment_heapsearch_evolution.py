import pickle
import os
import torch 
import logging
import argparse
import csv
import numpy as np

from dreamcoder_dataset_loader import filter_tasks_for_model, load_tasks
from experiment_helper import task_set2dataset, task_set2uniform_dataset
from run_experiment import gather_data
from type_system import Arrow, List, INT
from model_loader import build_dreamcoder_intlist_model, get_model_name

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()
verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

# ========================================================================
# Parameters
# ========================================================================
save_folder = "."
nb_epochs: int = 100
test_network_at = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
ADD_SOLUTIONS = False
# ========================================================================



# Make sure we use a model from scratch
cur_dsl, cfg, model = build_dreamcoder_intlist_model(autoload=False)
type_request = Arrow(List(INT), List(INT))
tasks = load_tasks()
tasks = filter_tasks_for_model(tasks, model)




# Split train / test
np.random.seed(0)

# Compute solutions
solutions_file = "./solutions_dreamcoder.pickle"
solutions = {}
if os.path.exists(solutions_file):
    with open(solutions_file, "rb") as fd:
        solutions = pickle.load(fd)

print("Computing solutions...")
all_indices = list(range(len(tasks)))
# Load solutions
kept_indices = [i for i, (name, ios) in enumerate(tasks) if name in solutions]
for el in kept_indices:
    all_indices.remove(el)

if ADD_SOLUTIONS:
    print("\t", len(all_indices), "remaining tasks to solve...")
    # Get corresponding tasks
    remaining_tasks = [t for i, t in enumerate(
        tasks) if i in all_indices]
    # Try to solve tasks
    dataset = task_set2uniform_dataset(remaining_tasks, cur_dsl)
    data = gather_data(dataset, 0)
    # Update those who fit into test or train set
    for i, (name, el) in enumerate(data):
        if el[0] is not None:
            solutions[name] = el[0]
            kept_indices.append(all_indices[i])
    # Save
    with open(solutions_file, "wb") as fd:
        pickle.dump(solutions, fd)
    print("Now relaunch using different timeout and max_programs in run_experiment.py")
    os._exit(0)

tasks = [(name, ios) for name, ios in tasks if name in solutions]
EVAL_TASKS = tasks

model_name = get_model_name(model)
filename = f"{save_folder}/algo_Heap Search - Uniform_model_{model_name}_dataset_dreamcoder_gen_results_semantic.csv"


if not os.path.exists(filename):
    with torch.no_grad():
        dataset = task_set2uniform_dataset(EVAL_TASKS, cur_dsl)

    data = gather_data(dataset, 0)
    rows = [[el[0]] + list(el[1]) for el in data]
    with open(filename, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["task_name", "program", "search_time", "evaluation_time",
                        "nb_programs", "cumulative_probability", "probability"])
        writer.writerows(rows)

############################
######## TRAINING ##########
############################

def test_network(model, i):
    model_name = get_model_name(model)
    filename = f"{save_folder}/algo_Heap Search - T={i}_model_{model_name}_dataset_dreamcoder_gen_results_semantic.csv"

    print("\tgenerating grammars...")
    with torch.no_grad():
        dataset = task_set2dataset(EVAL_TASKS, model, cur_dsl)

    print("\tgathering data...")
    data = gather_data(dataset, 0)
    rows = [[el[0]] + list(el[1]) for el in data]
    with open(filename, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["task_name", "program", "search_time", "evaluation_time",
                         "nb_programs", "cumulative_probability", "probability"])
        writer.writerows(rows)
    
    print("\tdone with ", len([1 for _, d in data if d[0] is not None]), "solved")



def train(model):

    batch_IOs, batch_program = [], []
    for name, ios in tasks:
        prog = solutions[name]
        batch_IOs.append([([i[0]], o) for i, o in ios])
        batch_program.append(model.ProgramEncoder(prog))

    for i in range(nb_epochs):
        if i in test_network_at:
            print("Testing...")
            test_network(model, i)
        
        model.optimizer.zero_grad()
        batch_predictions = model(batch_IOs)
        loss_value = model.loss(
                batch_predictions, torch.stack(batch_program))
        loss_value.backward()
        model.optimizer.step()
        print("\tminibatch: {}\t loss: {} metrics: {}".format(i, float(loss_value), model.metrics(loss=float(loss_value), batch_size=len(batch_IOs))))
    test_network(model, i + 1)


train(model)
