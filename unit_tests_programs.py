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

from Algorithms.a_star import a_star

class TestSum(unittest.TestCase):
    def test_programs(self):
        """
        Checks the evaluation of programs
        """
        p1 = BasicPrimitive("MAP")
        p2 = BasicPrimitive("MAP", type_=PolymorphicType(name="test"))

        # checking whether they represent the same programs and same types
        self.assertTrue(repr(p1) == repr(p2))
        self.assertTrue(p1.typeless_eq(p2))
        self.assertFalse(p1.__eq__(p2))
        self.assertFalse(id(p1) == id(p2))

        t0 = PolymorphicType("t0")
        t1 = PolymorphicType("t1")
        semantics = {
            "+1": lambda x: x + 1,
            "MAP": lambda f: lambda l: list(map(f, l)),
        }
        primitive_types = {
            "+1": Arrow(INT, INT),
            "MAP": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
        }
        toy_DSL = DSL(semantics, primitive_types)

        p0 = Function(BasicPrimitive("+1"), [Variable(0)])
        env = (2, None)
        self.assertTrue(p0.eval(toy_DSL, env, 0) == 3)

        p1 = Function(BasicPrimitive("MAP"), [BasicPrimitive("+1"), Variable(0)])
        env = ([2, 4], None)
        self.assertTrue(p1.eval(toy_DSL, env, 0) == [3, 5])

    def test_evaluation_from_compressed(self):
        """
        Check if evaluation_from_compressed evaluates correctly the programs
        """
        N = 20_000  # we test against the first N programs

        deepcoder = DSL(semantics, primitive_types)
        type_request = Arrow(List(INT), List(INT))
        deepcoder_CFG = deepcoder.DSL_to_CFG(type_request)
        deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG()

        gen_a_star = a_star(deepcoder_PCFG)

        environment = ([2, 3, 1], None)

        r = type_request.returns()
        for i in range(N):
            program_compressed = next(gen_a_star)
            program = reconstruct_from_compressed(program_compressed, r)
            program_as_list = []
            eval_from_compressed = evaluation_from_compressed(
                program_compressed, deepcoder, environment, r
            )
            eval_from_program = program.eval_naive(deepcoder, environment)
            self.assertEqual(eval_from_compressed, eval_from_program)

if __name__ == "__main__":
    unittest.main(verbosity=2)
