from type_system import BOOL, INT, STRING, Arrow, Type
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


def filter_examples(examples, nb_arguments_max, max_list_size, lexicon, verbose=False):
    filtered_examples = []
    for i, o in examples:

        if len(i) - 1 > nb_arguments_max:
            if verbose:
                print("\ttoo many arguments:", len(i) - 1, ">", nb_arguments_max)
            continue
        li = [x for x in i if hasattr(x, "__len__")]
        nli = [x for x in i if not hasattr(x, "__len__")]
        # List input
        if any(len(x) > max_list_size for x in li):
            if verbose:
                print("\tinput iterable too long:", max(len(x) for x in li), ">", max_list_size)
            continue
        if any(any(el not in lexicon for el in x) for x in li):
            if verbose:
                print("\tinput iterable not in lexicon:", [
                    [el for el in x if el not in lexicon] for x in li])
            continue
        # List output
        if hasattr(o, "__len__") and len(o) > max_list_size:
            if verbose:
                print("\toutput iterable too long:", len(o), ">", max_list_size)
            continue
        if hasattr(o, "__len__") and any(x not in lexicon for x in o):
            if verbose:
                print("\toutput iterable not in lexicon:", 
                    [el for el in o if el not in lexicon])
            continue
        # Non list input
        if any(x not in lexicon and x is not None for x in nli):
            if verbose:
                print("\tinput not in lexicon:", [x for x in nli if x not in lexicon])
            continue
        # Non list output
        if not hasattr(o, "__len__") and o not in lexicon:
            if verbose:
                print("\toutput not in lexicon:", o)
            continue
        filtered_examples.append((i, o))
    return filtered_examples   


def __get_type__(el) -> Type:
    if isinstance(el, int):
        return INT
    elif isinstance(el, str):
        return STRING
    return type_system.List(INT)


def __get_type_request(examples):
    input, output = examples[0]
    type_req = __get_type__(output)
    for el in input[:-1]:
        type_req = Arrow(__get_type__(el), type_req)
    return type_req
