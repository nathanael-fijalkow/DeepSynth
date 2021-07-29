import logging
import unittest
import random
from math import sqrt
from scipy.stats import chisquare

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL, STRING
from program import Program, Function, Variable, BasicPrimitive, New
from program_as_list import evaluation_from_compressed, reconstruct_from_compressed

from dsl import DSL
from DSL.deepcoder import semantics,primitive_types

class TestSum(unittest.TestCase):
    def test_construction_CFG_toy(self):
        """
        Checks the construction of a CFG from a toy DSL
        """
        t0 = PolymorphicType("t0")
        t1 = PolymorphicType("t1")
        semantics = {
            "RANGE": (),
            "HEAD": (),
            "SUCC": (),
            "MAP": (),
        }
        primitive_types = {
            "HEAD": Arrow(List(INT), INT),
            "RANGE": Arrow(INT, List(INT)),
            "SUCC": Arrow(INT, INT),
            "MAP": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
        }
        toy_DSL = DSL(semantics, primitive_types)
        type_request = Arrow(List(INT), List(INT))
        toy_CFG = toy_DSL.DSL_to_CFG(type_request)
        self.assertTrue(len(toy_CFG.rules) == 14)
        self.assertTrue(len(toy_CFG.rules[toy_CFG.start]) == 3)

    def test_construction_CFG_deepcoder(self):
        """
        Checks the construction of a PCFG from the DeepCoder DSL
        """
        deepcoder = DSL(semantics, primitive_types)
        type_request = Arrow(List(INT), List(INT))
        deepcoder_CFG = deepcoder.DSL_to_CFG(type_request)

        # checks that all non-terminals are productive
        for S in deepcoder_CFG.rules:
            self.assertTrue(len(deepcoder_CFG.rules[S]) > 0)
            for P in deepcoder_CFG.rules[S]:
                args_P = deepcoder_CFG.rules[S][P]
                for arg in args_P:
                    self.assertTrue(arg in deepcoder_CFG.rules)

    def test_construction_PCFG_toy(self):
        """
        Checks the construction of a PCFG from a DSL
        """
        t0 = PolymorphicType("t0")
        t1 = PolymorphicType("t1")
        semantics = {
            "RANGE": (),
            "HEAD": (),
            "TAIL": (),
            "SUCC": (),
            "PRED": (),
            "MAP": (),
        }
        primitive_types = {
            "HEAD": Arrow(List(INT), INT),
            "TAIL": Arrow(List(INT), INT),
            "RANGE": Arrow(INT, List(INT)),
            "SUCC": Arrow(INT, INT),
            "PRED": Arrow(INT, INT),
            "MAP": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
        }
        toy_DSL = DSL(semantics, primitive_types)
        type_request = Arrow(List(INT), List(INT))
        toy_CFG = toy_DSL.DSL_to_CFG(type_request)
        toy_PCFG = toy_CFG.CFG_to_Uniform_PCFG()
        toy_PCFG.compute_max_probability()

        max_program = Function(
            BasicPrimitive("MAP"),
            [
                BasicPrimitive("HEAD"),
                Function(BasicPrimitive("MAP"), [BasicPrimitive("RANGE"), Variable(0)]),
            ],
        )

        self.assertTrue(
            toy_PCFG.max_probability[toy_PCFG.start].typeless_eq(max_program)
        )

        for S in toy_PCFG.rules:
            max_program = toy_PCFG.max_probability[S]
            self.assertTrue(
                max_program.probability[(toy_PCFG.__hash__(), S)]
                == toy_PCFG.probability_program(S, max_program)
            )

    def test_construction_PCFG_deepcoder(self):
        """
        Checks the construction of a PCFG from a DSL
        """
        deepcoder = DSL(semantics, primitive_types)
        type_request = Arrow(List(INT), List(INT))
        deepcoder_CFG = deepcoder.DSL_to_CFG(type_request)
        deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()
        deepcoder_PCFG.compute_max_probability()

        for S in deepcoder_PCFG.rules:
            max_program = deepcoder_PCFG.max_probability[S]
            self.assertTrue(
                deepcoder_PCFG.max_probability[S].probability[
                    (deepcoder_PCFG.__hash__(), S)
                ]
                == deepcoder_PCFG.probability_program(S, max_program)
            )
            for P in deepcoder_PCFG.rules[S]:
                max_program = deepcoder_PCFG.max_probability[(S, P)]
                self.assertTrue(
                    deepcoder_PCFG.max_probability[(S, P)].probability[
                        (deepcoder_PCFG.__hash__(), S)
                    ]
                    == deepcoder_PCFG.probability_program(S, max_program)
                )

if __name__ == "__main__":
    unittest.main(verbosity=2)
