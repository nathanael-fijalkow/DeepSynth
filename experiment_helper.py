from type_system import BOOL, INT, STRING, Arrow
import type_system
from Predictions.models import RulesPredictor, BigramsPredictor
from pcfg import PCFG
from typing import Callable, List, Tuple
from dsl import DSL
from program import Program

def make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], bool]:
    def checker(program: Program, use_cached_evaluator: bool) -> bool:
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                out = program.eval(dsl, input, i)
                if output != out:
                    return False
            return True
        else:
            for example in examples:
                input, output = example
                out = program.eval_naive(dsl, input)
                if output != out:
                    return False
            return True
    return checker


def make_program_checker_with_constants(dsl: DSL, examples, constants) -> Callable[[Program, bool], Tuple[bool, Program]]:
    def checker(program: Program, use_cached_evaluator: bool) -> Tuple[bool, Program]:
        programs = program.make_all_constant_variations(constants)
        for program in programs:
            failed = False
            if use_cached_evaluator:
                for i, example in enumerate(examples):
                    input, output = example
                    out = program.eval(dsl, input, i)
                    if output != out:
                        failed = True
                        break
            else:
                for example in examples:
                    input, output = example
                    out = program.eval_naive(dsl, input)
                    if output != out:
                        failed = True
                        break
            if not failed:
                return True, program
        return False, None
    return checker

def task_set2dataset(tasks, model, dsl: DSL) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]:
    dataset = []
    for task in tasks:
        if len(task) == 3:
            name, examples, constants = task
        else:
            name, examples = task
            constants = None
        try:
            ex = [[([i[0]], o) for i, o in examples]]
            grammar = model(ex)[0]
        except AssertionError as e:
            print("experiment_helper.py: task_set2dataset: An error occured while generating a grammar for task:", name, "\n\t", e)
            continue
        if isinstance(model, RulesPredictor):
            grammar = model.reconstruct_grammars([grammar])[0]
        if isinstance(model, BigramsPredictor):
            grammar = model.reconstruct_grammars(
                [grammar], [__get_type_request(examples)], tensors=False)[0]
            grammar = grammar.normalise()

        dataset.append(
            (name, grammar, make_program_checker_with_constants(dsl, examples, constants) if constants else make_program_checker(dsl, examples)))
    return dataset


def filter_examples(examples, nb_arguments_max, max_list_size, lexicon):
    filtered_examples = []
    for i, o in examples:
        if len(i) > nb_arguments_max:
            continue
        # List input
        if any(hasattr(x, "__len__") and len(x) > max_list_size for x in i):
            continue
        if any(hasattr(x, "__len__") and any(el not in lexicon for el in x) for x in i):
            continue
        # List output
        if hasattr(o, "__len__") and len(o) > max_list_size:
            continue
        if hasattr(o, "__len__") and any(x not in lexicon for x in o):
            continue
        # Non list input
        if any(not hasattr(x, "__len__") and x not in lexicon and x is not None for x in i):
            continue
        # Non list output
        if not hasattr(o, "__len__") and o not in lexicon:
            continue
        filtered_examples.append((i, o))
    return filtered_examples   

def __get_type_request(examples):
    input, output = examples[0]
    type_req = INT if isinstance(output, int) else (STRING if isinstance(output, str) else type_system.List(INT))
    for el in input:
        if isinstance(el, int):
            type_req = Arrow(INT, type_req)
        elif isinstance(el, str):
            type_req = Arrow(STRING, type_req)
        else:
            type_req = Arrow(type_system.List(INT), type_req)
    return type_req
