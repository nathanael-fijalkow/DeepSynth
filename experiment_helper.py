from type_system import INT, STRING, Arrow, Type
import type_system
from Predictions.models import RulesPredictor, BigramsPredictor
from pcfg import PCFG
from typing import Callable, List, Tuple
from dsl import DSL
from program import Program

def make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], bool]:
    def checker(prog: Program, use_cached_evaluator: bool) -> bool:
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                out = prog.eval(dsl, input, i)
                if output != out:
                    return False
            return True
        else:
            for example in examples:
                input, output = example
                out = prog.eval_naive(dsl, input)
                if output != out:
                    return False
            return True
    return checker


def make_program_checker_with_constants(dsl: DSL, examples, constants) -> Callable[[Program, bool], Tuple[bool, Program]]:
    def checker(prog: Program, use_cached_evaluator: bool) -> Tuple[bool, Program]:
        programs = prog.make_all_constant_variations(constants)
        for fixed_prog in programs:
            failed = False
            if use_cached_evaluator:
                for i, example in enumerate(examples):
                    input, output = example
                    out = fixed_prog.eval(dsl, input, i)
                    if output != out:
                        failed = True
                        break
            else:
                for example in examples:
                    input, output = example
                    out = fixed_prog.eval_naive(dsl, input)
                    if output != out:
                        failed = True
                        break
            if not failed:
                return True, fixed_prog
        return False, None
    return checker

def task_set2dataset(tasks, model, dsl: DSL) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]:
    dataset = []
    batch_IOs = []
    batch_types = []
    # Prepare batch
    for task in tasks:
        if len(task) == 3:
            name, examples, constants = task
        else:
            name, examples = task
            constants = None
        ex = [([i[0]], o) for i, o in examples]
        batch_IOs.append(ex)
        if isinstance(model, BigramsPredictor):
            batch_types.append(__get_type_request(examples))
    # Inference
    try:
        grammars = model(batch_IOs)
    except AssertionError as e:
        print("experiment_helper.py: task_set2dataset: An error occured while generating grammars:\n\t", e)
        return []
    # Reconstruction
    if isinstance(model, RulesPredictor):
        grammars = model.reconstruct_grammars(grammars)
    if isinstance(model, BigramsPredictor):
        grammars = model.reconstruct_grammars(
            grammars, batch_types, tensors=False)
        grammars = [g.normalise() for g in grammars]
    # To dataset
    for i, grammar in enumerate(grammars):
        name = tasks[i][0]
        examples = tasks[i][1]
        constants = None if len(tasks[i]) < 3 else tasks[i][2]
        dataset.append(
            (name, grammar, make_program_checker_with_constants(dsl, examples, constants) if constants else make_program_checker(dsl, examples)))
    return dataset


def filter_examples(examples, nb_arguments_max, max_list_size, lexicon, verbose=False):
    filtered_examples = []
    one_output_is_nonempty = False
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
        if not hasattr(o, "__len__") or len(o) > 0:
            one_output_is_nonempty = True
        filtered_examples.append((i, o))
    if one_output_is_nonempty:
        return filtered_examples   
    return []


def __get_type__(el) -> Type:
    if isinstance(el, int):
        return INT
    elif isinstance(el, str):
        return STRING
    return type_system.List(INT)


def __get_type_request(examples):
    input, output = examples[0]
    type_req = __get_type__(output)
    for el in input[:-1][::-1]:
        type_req = Arrow(__get_type__(el), type_req)
    return type_req
