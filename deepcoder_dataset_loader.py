from Predictions.models import RulesPredictor
import json
from type_system import BOOL, INT, Arrow, List, Type
from typing import Any, Tuple
import typing

def load_tasks(file: str) -> typing.List[Tuple[str,Any]]:
    tasks = []
    with open(file, "r") as fd:
        raw_tasks = json.load(fd)
        for raw_task in raw_tasks:
            name = raw_task["program"]
            raw_examples = raw_task["examples"]
            examples = [((raw_example["inputs"][0], None), raw_example["output"])
                        for raw_example in raw_examples]
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
    return Arrow(__get_type(input[0], [i[0] for i, _ in examples[1:]]), __get_type(output, [o for _, o in examples[1:]]))


def filter_tasks_for_model(tasks, model) -> typing.List[Tuple[str, Any]]:
    filtered_tasks = []
    for task in tasks:
        name, examples = task
        
        # Remove tasks that return null
        if any(o is None for _, o in examples):
            continue
        try:
            type_request: Type = __get_type_request(examples)
        except:
            # Skip tasks where the solution is to always return an empty list
            continue
        if isinstance(model, RulesPredictor) and type_request != Arrow(List(INT), List(INT)):
            continue


        examples = [(i, o) for i, o in examples if len(i[0]) <= model.IOEncoder.size_max and all(
            [el in model.IOEncoder.symbolToIndex for el in i[0]]) and all([el in model.IOEncoder.symbolToIndex for el in o])]
        if len(examples) == 0:
            continue

        filtered_tasks.append((name, examples))
    return filtered_tasks
