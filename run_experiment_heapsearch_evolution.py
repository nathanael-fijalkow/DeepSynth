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
nb_epochs: int = 10
test_network_at = [0, 1, 2, 4, 7]
test_split = 0.4
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
train_indices = []
test_indices = []
remaining_to_select = int(len(tasks) * (1 - test_split)) - len(solutions)
added_solutions = False
# Load solutions
train_indices = [i for i, (name, ios) in enumerate(tasks) if name in solutions]
for el in train_indices:
    all_indices.remove(el)

while remaining_to_select > 0:
    added_solutions = True
    print("\t", remaining_to_select, "remaining tasks to solve...")
    # Select random indices
    remaining_train_indices = np.random.choice(
        all_indices, size=remaining_to_select, replace=False)
    # Remove them from the universe
    for el in remaining_train_indices:
        all_indices.remove(el)
    # Get corresponding tasks
    remaining_train_tasks = [t for i, t in enumerate(
        tasks) if i in remaining_train_indices]
    # Try to solve tasks
    dataset = task_set2uniform_dataset(remaining_train_tasks, cur_dsl)
    data = gather_data(dataset, 0)
    # Update those who fit into test or train set
    for i, (name, el) in enumerate(data):
        if el[0] is None:
            test_indices.append(remaining_train_indices[i])
        else:
            solutions[name] = el[0]
            train_indices.append(remaining_train_indices[i])
            remaining_to_select -= 1
    # Save
    with open(solutions_file, "wb") as fd:
        pickle.dump(solutions, fd)
if added_solutions:
    print("Succeeded!")
    with open(solutions_file, "wb") as fd:
        pickle.dump(solutions, fd)
    print("Now relaunch using different timeout and max_programs in run_experiment.py")
    os._exit(0)


train_tasks = [(name, ios) for name, ios in tasks if name in solutions]
test_tasks = [(name, ios) for name, ios in tasks if name not in solutions]

############################
######## TRAINING ##########
############################

def test_network(model, i, tasks):
    model_name = get_model_name(model)
    filename = f"{save_folder}/algo_Heap Search - T={i}_model_{model_name}_dataset_dreamcoder_gen_results_semantic.csv"

    print("\tgenerating grammars...")
    with torch.no_grad():
        dataset = task_set2dataset(tasks, model, cur_dsl)

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
    for name, ios in train_tasks:
        prog = solutions[name]
        batch_IOs.append([([i[0]], o) for i, o in ios])
        batch_program.append(model.ProgramEncoder(prog))

    for i in range(nb_epochs):
        if i in test_network_at:
            print("Testing...")
            test_network(model, i, test_tasks)
        
        model.optimizer.zero_grad()
        batch_predictions = model(batch_IOs)
        loss_value = model.loss(
                batch_predictions, torch.stack(batch_program))
        loss_value.backward()
        model.optimizer.step()
        print("\tminibatch: {}\t loss: {} metrics: {}".format(i, float(loss_value), model.metrics(loss=float(loss_value), batch_size=len(batch_IOs))))
    test_network(model, i, tasks)


train(model)
