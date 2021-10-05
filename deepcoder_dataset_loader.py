from Predictions.models import RulesPredictor
import json
from program import BasicPrimitive, Function, Variable
from type_system import BOOL, INT, Arrow, List, Type
from typing import Any, Tuple
import typing
from DSL import deepcoder
import dsl

my_dsl = dsl.DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

def load_tasks(file: str) -> Tuple[typing.List[Tuple[str,Any]], set]:
    tasks = []
    all_types = set()
    with open(file, "r") as fd:
        raw_tasks = json.load(fd)
        for raw_task in raw_tasks:
            name = raw_task["program"]
            raw_examples = raw_task["examples"]
            examples = [((raw_example["inputs"][0], None), raw_example["output"])
                        for raw_example in raw_examples]

            prog, type_request = __str2prog(name)
            tasks.append((prog, examples))
            all_types.add(type_request)

    return tasks, all_types


def __str2prog(s: str):
    parts = s.split("|")
    stack = []
    var = 0
    type_stack = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        if name == "LIST":
            stack.append(Variable(var, List(INT)))
            var += 1
            type_stack.append(List(INT))
            continue
        if name == "INT":
            stack.append(Variable(var, INT))
            var += 1
            type_stack.append(INT)
            continue
        if name not in deepcoder.primitive_types:
            name = name + "[" + subparts.pop(0) + "]"
        primitive = BasicPrimitive(name, deepcoder.primitive_types[name])
        targets = [int(x) for x in subparts]
        arguments = [stack[x] for x in targets]
        stack.append(Function(primitive, arguments, type_=primitive.type.returns()))
    type_request = stack[-1].type
    while type_stack:
        type_request = Arrow(type_stack.pop(), type_request)
    return stack[-1], type_request


def filter_tasks_for_model(tasks, model) -> typing.List[Tuple[str, Any]]:
    filtered_tasks = []
    for task in tasks:
        name, examples = task
        
        # Remove tasks that return null
        if any(o is None for _, o in examples):
            continue
        type_request: Type = name.type
        if isinstance(model, RulesPredictor) and type_request != Arrow(List(INT), List(INT)):
            continue


        examples = [(i, o) for i, o in examples if (len(i[0]) <= model.IOEncoder.size_max and all(
            [el in model.IOEncoder.symbolToIndex for el in i[0]])) and ((isinstance(o, int) and o in model.IOEncoder.symbolToIndex) or all([el in model.IOEncoder.symbolToIndex for el in o]))]
        if len(examples) == 0:
            continue

        filtered_tasks.append((name, examples))
    return filtered_tasks
