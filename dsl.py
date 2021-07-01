from type_system import *
from program import *
from cfg import *
from pcfg import *

from collections import deque
import copy
import time


class DSL:
    """
    Object that represents a domain specific language

    list_primitives: a list of primitives, either BasicPrimitive or New

    semantics: a dictionary of the form {P : f}
    mapping a program P to its semantics f
    for P a BasicPrimitive
    """

    def __init__(self, semantics, primitive_types):
        self.list_primitives = []
        self.semantics = {}

        for p in primitive_types:
            formatted_p = format(p)
            if formatted_p in semantics:
                self.semantics[formatted_p] = semantics[formatted_p]
                P = BasicPrimitive(
                    primitive=formatted_p, type_=primitive_types[p], probability={}
                )
                self.list_primitives.append(P)
            else:
                P = New(body=p.body, type_=primitive_types[p], probability={})
                self.list_primitives.append(P)

    def __repr__(self):
        s = "Print a DSL\n"
        for P in self.list_primitives:
            s = s + "{}: {}\n".format(P, P.type)
        return s

    def instantiate_polymorphic_types(self, upper_bound_type_size=10):
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types_P, set_polymorphic_types_P = P.type.decompose_type()
            set_basic_types = set_basic_types | set_basic_types_P

        # print("basic types", set_basic_types)

        set_types = set(set_basic_types)
        for type_ in set_basic_types:
            new_type = List(type_)
            set_types.add(new_type)
            new_type = List(new_type)
            set_types.add(new_type)

        for type_ in set_basic_types:
            for type_2 in set_basic_types:
                new_type2 = Arrow(type_, type_2)
                set_types.add(new_type2)

        # print("set_types", set_types)

        new_primitive_types = {}

        for P in self.list_primitives:
            assert isinstance(P, (New, BasicPrimitive))
            type_P = P.type
            set_basic_types_P, set_polymorphic_types_P = type_P.decompose_type()
            if set_polymorphic_types_P:
                set_instantiated_types = set()
                set_instantiated_types.add(type_P)
                for poly_type in set_polymorphic_types_P:
                    new_set_instantiated_types = set()
                    for type_ in set_types:
                        for instantiated_type in set_instantiated_types:
                            unifier = {str(poly_type): type_}
                            intermediate_type = copy.deepcopy(instantiated_type)
                            new_type = intermediate_type.apply_unifier(unifier)
                            if new_type.size() <= upper_bound_type_size:
                                new_set_instantiated_types.add(new_type)
                    set_instantiated_types = new_set_instantiated_types
                for type_ in set_instantiated_types:
                    if isinstance(P, New):
                        instantiated_P = New(P.body, type_, probability={})
                    if isinstance(P, BasicPrimitive):
                        instantiated_P = BasicPrimitive(
                            P.primitive, type_, probability={}
                        )
                    self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

    def DSL_to_CFG(
        self,
        type_request,
        upper_bound_type_size=10,
        max_program_depth=4,
        min_variable_depth=1,
        n_gram=1,
    ):
        """
        Constructs a CFG from a DSL imposing bounds on size of the types
        and on the maximum program depth
        """
        self.instantiate_polymorphic_types(upper_bound_type_size)

        return_type = type_request.returns()
        args = type_request.arguments()

        rules = {}

        def repr(current_type, context, depth):
            if len(context) == 0:
                return current_type, None, depth
            if n_gram == 1:
                return current_type, context[0], depth
            return current_type, context, depth

        list_to_be_treated = deque()
        list_to_be_treated.append((return_type, [], 0))

        while len(list_to_be_treated) > 0:
            current_type, context, depth = list_to_be_treated.pop()
            non_terminal = repr(current_type, context, depth)

            # a non-terminal is a triple (type, context, depth)
            # if n_gram = 0 context = None
            # otherwise context is a list of (primitive, number_argument)
            # print("\ncollecting from the non-terminal ", non_terminal)

            if non_terminal not in rules:
                rules[non_terminal] = {}

            if depth < max_program_depth and depth >= min_variable_depth:
                for i in range(len(args)):
                    if current_type == args[i]:
                        var = Variable(i, current_type, probability={})
                        rules[non_terminal][var] = []

            if depth == max_program_depth - 1:
                for P in self.list_primitives:
                    type_P = P.type
                    return_P = type_P.returns()
                    if return_P == current_type and len(type_P.arguments()) == 0:
                        rules[non_terminal][P] = []

            elif depth < max_program_depth:
                for P in self.list_primitives:
                    type_P = P.type
                    arguments_P = type_P.ends_with(current_type)
                    if arguments_P != None:
                        decorated_arguments_P = []
                        for i, arg in enumerate(arguments_P):
                            new_context = context.copy()
                            new_context = [(P, i)] + new_context
                            if len(new_context) > n_gram:
                                new_context.pop()
                            decorated_arguments_P.append(
                                repr(arg, new_context, depth + 1)
                            )
                            if (arg, new_context, depth + 1) not in list_to_be_treated:
                                list_to_be_treated.appendleft(
                                    (arg, new_context, depth + 1)
                                )

                        rules[non_terminal][P] = decorated_arguments_P

        # print(rules)
        return CFG(
            start=(return_type, None, 0),
            rules=rules,
            max_program_depth=max_program_depth,
        )

    def DSL_to_Uniform_PCFG(
        self,
        type_request,
        upper_bound_type_size=10,
        max_program_depth=4,
        min_variable_depth=1,
        n_gram=1,
    ):
        CFG = self.DSL_to_CFG(
            type_request,
            upper_bound_type_size,
            max_program_depth,
            min_variable_depth,
            n_gram,
        )
        augmented_rules = {}
        for S in CFG.rules:
            augmented_rules[S] = {}
            p = len(CFG.rules[S])
            for P in CFG.rules[S]:
                augmented_rules[S][P] = (CFG.rules[S][P], 1 / p)
        return PCFG(
            start=CFG.start, rules=augmented_rules, max_program_depth=max_program_depth
        )

    def DSL_to_Random_PCFG(
        self,
        type_request,
        upper_bound_type_size=10,
        max_program_depth=4,
        min_variable_depth=1,
        n_gram=1,
        alpha=0.7,
    ):
        CFG = self.DSL_to_CFG(
            type_request,
            upper_bound_type_size,
            max_program_depth,
            min_variable_depth,
            n_gram,
        )
        new_rules = {}
        for S in CFG.rules:
            out_degree = len(CFG.rules[S])
            # weights with alpha-exponential decrease
            weights = [random.random() * (alpha ** i) for i in range(out_degree)]
            s = sum(weights)
            # normalization
            weights = [w / s for w in weights]
            random_permutation = list(
                np.random.permutation([i for i in range(out_degree)])
            )
            new_rules[S] = {}
            for i, P in enumerate(CFG.rules[S]):
                new_rules[S][P] = (CFG.rules[S][P], weights[random_permutation[i]])
        return PCFG(
            start=CFG.start, rules=new_rules, max_program_depth=max_program_depth
        )
