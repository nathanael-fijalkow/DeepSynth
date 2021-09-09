from Predictions.models import GlobalRulesPredictor
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

def task_set2dataset(tasks, model, dsl) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]:
    dataset = []
    for task in tasks:
        name, examples = task
        try:
            ex = [[([i[0]], o) for i, o in examples]]
            grammar = model(ex)[0]
        except AssertionError:
            continue
        if isinstance(model, GlobalRulesPredictor):
            grammar = model.reconstruct_grammars([grammar])[0]

        dataset.append((name, grammar, make_program_checker(dsl, examples)))
    return dataset