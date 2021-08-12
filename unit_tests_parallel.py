import unittest
from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL, STRING
from program import Program, Function, Variable, BasicPrimitive, New
from program_as_list import evaluation_from_compressed, reconstruct_from_compressed

from dsl import DSL
from DSL.deepcoder import semantics,primitive_types

from Algorithms.sqrt_sampling import sqrt_sampling_with_sbsur
# from Algorithms.ray_parallel import make_parallel_pipelines, start

import grammar_splitter

class TestSum(unittest.TestCase):
 
    # TODO: Uncomment to test Ray
    # def test_ray_parallel_and_splitting(self):
    #     """
    #     Check if ray_parallel + splitting algorithm does not miss any program.
    #     """
    #     t0 = PolymorphicType("t0")
    #     t1 = PolymorphicType("t1")
    #     semantics = {
    #         "RANGE": (),
    #         "HEAD": (),
    #         "TAIL": (),
    #         "SUCC": (),
    #         "PRED": (),
    #         "MAP": (),
    #     }
    #     primitive_types = {
    #         "HEAD": Arrow(List(INT), INT),
    #         "TAIL": Arrow(List(INT), INT),
    #         "RANGE": Arrow(INT, List(INT)),
    #         "SUCC": Arrow(INT, INT),
    #         "PRED": Arrow(INT, INT),
    #         "MAP": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
    #     }
    #     toy_DSL = DSL(semantics, primitive_types)
    #     type_request = Arrow(List(INT), List(INT))
    #     deepcoder_CFG = toy_DSL.DSL_to_CFG(type_request)
    #     deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG(alpha=0.8)
    #     n = 100

    #     def insert_prefix(prefix, prog):
    #         try:
    #             head, tail = prog
    #             return (head, insert_prefix(prefix, tail))
    #         except:
    #             return prefix

    #     def bounded_generator(prefix, pcfg):
    #         # It should be easier for 
    #         def new_gen():
    #             for p in sqrt_sampling_with_sbsur(pcfg):
    #                 yield insert_prefix(prefix, p)
    #         return new_gen
    #     make_generators = [bounded_generator(
    #         prefix, pcfg) for prefix, pcfg in grammar_splitter.split(deepcoder_PCFG, 10)]
    #     make_filter = lambda: lambda x: True

    #     producers, filters, _, out = make_parallel_pipelines(make_generators, make_filter, 2, 1000, 10000, 10)
    #     start(producers)
    #     start(filters)

    #     seen_programs = set()
    #     while len(seen_programs) < n:
    #         program = out.get()
    #         # prog = reconstruct_from_compressed(program, r)
    #         self.assertNotIn(program, seen_programs)
    #         seen_programs.add(program)


    def test_splitter(self):
        """
        Check if the grammar splitter generate disjoint valid grammars
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
        deepcoder_CFG = toy_DSL.DSL_to_CFG(type_request)
        deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG(alpha=0.8)

        r = type_request.returns()

        n = 10_0000
        splits = 4
        programs_per_split = n // splits


        def insert_prefix(prefix, prog):
            try:
                head, tail = prog
                return (head, insert_prefix(prefix, tail))
            except:
                return prefix

        
        seen_programs = set()
        preceding_sets = set()
        for prefix, pcfg in grammar_splitter.split(deepcoder_PCFG, splits, alpha=1.05):
            j = 0
            # If there is one single program for this PCFG SBSUR won't generate it because there is no need for sampling
            if all([len(pcfg.rules[S]) == 1 for S in pcfg.rules]):
                continue
            for p in sqrt_sampling_with_sbsur(pcfg):
                cp = insert_prefix(prefix, p)
                program = reconstruct_from_compressed(cp, r)
                j += 1
                if j >= programs_per_split:
                    break
                self.assertNotIn(program, preceding_sets)
                self.assertNotIn(program, seen_programs)
                seen_programs.add(program)
                try:
                    p = deepcoder_PCFG.probability_program(deepcoder_PCFG.start, program)
                except:
                    self.assertFalse(True, "Failed computing probability")
            assert len(seen_programs) > 0
            preceding_sets = preceding_sets.union(seen_programs)
            seen_programs = set()

if __name__ == "__main__":
    unittest.main(verbosity=2)
