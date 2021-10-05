from collections import deque
import copy

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from program import Program, Function, Variable, BasicPrimitive, New
from cfg import CFG

from itertools import combinations_with_replacement

class DSL:
    """
    Object that represents a domain specific language

    list_primitives: a list of primitives, either BasicPrimitive or New

    semantics: a dictionary of the form {P : f}
    mapping a program P to its semantics f
    for P a BasicPrimitive
    """

    def __init__(self, semantics, primitive_types, no_repetitions=None):
        self.list_primitives = []
        self.semantics = {}
        self.no_repetitions = no_repetitions or set()

        for p in primitive_types:
            formatted_p = str(p)
            if formatted_p in semantics:
                self.semantics[formatted_p] = semantics[formatted_p]
                P = BasicPrimitive(
                    primitive=formatted_p, type_=primitive_types[p], probability={}
                )
                self.list_primitives.append(P)
            else:
                P = New(body=p.body, type_=primitive_types[p], probability={})
                self.list_primitives.append(P)
    
    def primitive_types(self):
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types_P = P.type.list_ground_types(polymorphic=False)
            for x in set_basic_types_P:
                set_basic_types.add(x)
        return list(set_basic_types)

    def return_types(self):
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types.add(P.type.returns())
        return list(set_basic_types)


    def all_type_requests(self, nb_arguments_max: int):
        types = self.primitive_types()
        all_return_types = self.return_types()
        requests = []
        for nb_arguments in range(1, nb_arguments_max + 1):
            signatures = combinations_with_replacement(types, nb_arguments)
            for signature in signatures:
                for return_type in all_return_types:
                    cur_type = return_type
                    for el in signature:
                        cur_type = Arrow(el, cur_type)
                    requests.append(cur_type)
        return requests

    def __str__(self):
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

        for P in self.list_primitives[:]:
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
        n_gram=2,
    ):
        """
        Constructs a CFG from a DSL imposing bounds on size of the types
        and on the maximum program depth
        """
        self.instantiate_polymorphic_types(upper_bound_type_size)

        return_type = type_request.returns()
        args = type_request.arguments()

        rules = {}

        def encode_non_terminal(current_type, context, depth):
            if len(context) == 0:
                return current_type, None, depth
            if n_gram == 2:
                return current_type, context[0], depth
            return current_type, context, depth

        list_to_be_treated = deque()
        list_to_be_treated.append((return_type, [], 0))

        while len(list_to_be_treated) > 0:
            current_type, context, depth = list_to_be_treated.pop()
            non_terminal = encode_non_terminal(current_type, context, depth)

            # a non-terminal is a triple (type, context, depth)
            # if n_gram == 1 then context = None
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
                    if isinstance(P, BasicPrimitive) and P.primitive in self.no_repetitions and \
                            context and len(context) > 0 and context[0][0].primitive == P.primitive:
                        continue
                    type_P = P.type
                    arguments_P = type_P.ends_with(current_type)
                    if arguments_P != None:
                        decorated_arguments_P = []
                        for i, arg in enumerate(arguments_P):
                            new_context = context.copy()
                            new_context = [(P, i)] + new_context
                            if len(new_context) > n_gram - 1:
                                new_context.pop()
                            decorated_arguments_P.append(
                                encode_non_terminal(arg, new_context, depth + 1)
                            )
                            if (arg, new_context, depth + 1) not in list_to_be_treated:
                                list_to_be_treated.appendleft(
                                    (arg, new_context, depth + 1)
                                )

                        rules[non_terminal][P] = decorated_arguments_P

        return CFG(
            start=(return_type, None, 0),
            rules=rules,
            max_program_depth=max_program_depth,
            clean=True
        )
