from Predictions.models import GlobalRulesPredictor
import glob
import pickle
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
        if isinstance(model, GlobalRulesPredictor) and type_request != Arrow(List(INT), List(INT)):
            continue

        examples = [(i, o) for i, o in examples if len(i[0]) <= model.IOEncoder.size_max and all(
            [el in model.IOEncoder.symbolToIndex for el in i[0]]) and all([el in model.IOEncoder.symbolToIndex for el in o])]
        if len(examples) == 0:
            continue

        filtered_tasks.append((name, examples))
    return filtered_tasks