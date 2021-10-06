from Predictions.models import RulesPredictor
import glob
import pickle
from experiment_helper import filter_examples
from type_system import BOOL, INT, Arrow, List, Type
from typing import Any, Tuple
import typing


def load_tasks(folder: str = "list_dataset") -> typing.List[Tuple[str,Any]]:
    # Load all tasks
    tasks = []
    for file in glob.glob(f"{folder}/*.pickle"):
        with open(file, "rb") as fd:
            (name, examples) = pickle.load(fd)
            tasks.append((name, examples))
    return tasks


def __get_type(el, fallback=None):
    if isinstance(el, bool):
        return BOOL
    elif isinstance(el, int):
        return INT
    elif isinstance(el, list):
        if len(el) > 0:
            return List(__get_type(el[0]))
        else:
            return __get_type(fallback[0], fallback[1:])
    elif isinstance(el, tuple):
        assert el[-1] == None
        return __get_type(el[0], el[1:-1])
    assert False, f"Unknown type for:{el}"


def __get_type_request(examples):
    input, output = examples[0]
    return Arrow(__get_type(input[0], [i[0] for i, o in examples[1:]]), __get_type(output, [o for i, o in examples[1:]]))


def filter_tasks_for_model(tasks, model) -> typing.List[Tuple[str, Any]]:
    filtered_tasks = []
    for task in tasks:
        name, examples = task
        type_request: Type = __get_type_request(examples)
        if isinstance(model, RulesPredictor) and type_request != Arrow(List(INT), List(INT)):
            continue

        examples = filter_examples(
            examples, model.IOEncoder.nb_arguments_max, model.IOEncoder.size_max, model.IOEncoder.symbolToIndex)
        if len(examples) == 0:
            continue

        filtered_tasks.append((name, examples))
    return filtered_tasks