import os
import sys

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType
from cons_list import index

from itertools import combinations_with_replacement

# dictionary { number of environment : value }
# environment: a cons list
# list = None | (value, list)
# probability: a dictionary {(G.__hash(), S) : probability}
# such that P.probability[S] is the probability that P is generated
# from the non-terminal S when the underlying PCFG is G.

# make sure hash is deterministic
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

class Program:
    """
    Object that represents a program: a lambda term with basic primitives
    """

    def __eq__(self, other):
        return (
            isinstance(self, Program)
            and isinstance(other, Program)
            and self.type.__eq__(other.type)
            and self.typeless_eq(other)
        )

    def typeless_eq(self, other):
        b = isinstance(self, Program) and isinstance(other, Program)
        b2 = (
            isinstance(self, Variable)
            and isinstance(other, Variable)
            and self.variable == other.variable
        )
        b2 = b2 or (
            isinstance(self, Function)
            and isinstance(other, Function)
            and self.function.typeless_eq(other.function)
            and len(self.arguments) == len(other.arguments)
            and all(
                [
                    x.typeless_eq(y)
                    for x, y in zip(self.arguments, other.arguments)
                ]
            )
        )
        b2 = b2 or (
            isinstance(self, Lambda)
            and isinstance(other, Lambda)
            and self.body.typeless_eq(other.body)
        )
        b2 = b2 or (
            isinstance(self, BasicPrimitive)
            and isinstance(other, BasicPrimitive)
            and self.primitive == other.primitive
        )
        b2 = b2 or (
            isinstance(self, New)
            and isinstance(other, New)
            and (self.body).typeless_eq(other.body)
        )
        return b and b2

    def __gt__(self, other):
        True

    def __lt__(self, other):
        False

    def __ge__(self, other):
        True

    def __le__(self, other):
        False

    def __hash__(self):
        return self.hash

    def is_constant(self):
        return True

    def derive_with_constants(self, constants):
        return self

    def make_all_constant_variations(self, constants_list):
        n_constants = self.count_constants()
        if n_constants == 0:
            return [self]
        all_possibilities = combinations_with_replacement(constants_list, n_constants)
        return [self.derive_with_constants(possibility) for possibility in all_possibilities]
        

    def count_constants(self):
        return 0

class Variable(Program):
    def __init__(self, variable, type_=UnknownType(), probability={}):
        # assert isinstance(variable, int)
        self.variable = variable
        # assert isinstance(type_, Type)
        self.type = type_
        self.hash = variable

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return "var" + format(self.variable)

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = index(environment, self.variable)
            self.evaluation[i] = result
            return result
        except (IndexError, ValueError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = index(environment, self.variable)
            return result
        except (IndexError, ValueError, TypeError):
            return None

    def is_constant(self):
        return False

class Function(Program):
    def __init__(self, function, arguments, type_=UnknownType(), probability={}):
        # assert isinstance(function, Program)
        self.function = function
        # assert isinstance(arguments, list)
        self.arguments = arguments
        self.type = type_
        self.hash = hash(tuple([arg.hash for arg in self.arguments] + [self.function.hash]))

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        if len(self.arguments) == 0:
            return format(self.function)
        else:
            s = "(" + format(self.function)
            for arg in self.arguments:
                s += " " + format(arg)
            return s + ")"

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            if len(self.arguments) == 0:
                return self.function.eval(dsl, environment, i)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval(dsl, environment, i)
                    evaluated_arguments.append(e)
                result = self.function.eval(dsl, environment, i)
                for evaluated_arg in evaluated_arguments:
                    result = result(evaluated_arg)
                self.evaluation[i] = result
                return result
        except (IndexError, ValueError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            if len(self.arguments) == 0:
                return self.function.eval_naive(dsl, environment)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval_naive(dsl, environment)
                    evaluated_arguments.append(e)
                result = self.function.eval_naive(dsl, environment)
                for evaluated_arg in evaluated_arguments:
                    result = result(evaluated_arg)
                return result
        except (IndexError, ValueError, TypeError):
            return None

    def is_constant(self):
        return all([self.function.is_constant()] + [arg.is_constant() for arg in self.arguments])

    def count_constants(self):
        return self.function.count_constants() + sum([arg.count_constants() for arg in self.arguments])

    def derive_with_constants(self, constants):
        return Function(self.function.derive_with_functions(constants), [argument.derive_with_functions(constants) for argument in self.arguments], self.type, self.probability)

class Lambda(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        # assert isinstance(body, Program)
        self.body = body
        # assert isinstance(type_, Type)
        self.type = type_
        self.hash = hash(94135 + body.hash)

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        s = "(lambda " + format(self.body) + ")"
        return s

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = lambda x: self.body.eval(dsl, (x, environment), i)
            self.evaluation[i] = result
            return result
        except (IndexError, ValueError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = lambda x: self.body.eval_naive(dsl, (x, environment))
            return result
        except (IndexError, ValueError, TypeError):
            return None

class BasicPrimitive(Program):
    def __init__(self, primitive, type_=UnknownType(), probability={}, constant_evaluation=None):
        # assert isinstance(primitive, str)
        self.primitive = primitive
        # assert isinstance(type_, Type)
        self.type = type_
        self.is_a_constant = not isinstance(type_, Arrow)
        self.constant_evaluation = constant_evaluation
        self.hash = hash(primitive) + self.type.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        """
        representation without type
        """
        return format(self.primitive)

    def eval(self, dsl, environment, i):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def eval_naive(self, dsl, environment):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def count_constants(self):
        return 1 if self.is_a_constant else 0

    def derive_with_constants(self, constants):
        if self.is_a_constant:
            value = constants.pop()
            return BasicPrimitive(self.primitive, self.type, self.probability, value)
        else:
            return self


class New(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        self.body = body
        self.type = type_
        self.hash = hash(783712 + body.hash) + type_.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return format(self.body)

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = self.body.eval(dsl, environment, i)
            self.evaluation[i] = result
            return result
        except (IndexError, ValueError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = self.body.eval_naive(dsl, environment)
            return result
        except (IndexError, ValueError, TypeError):
            return None

    def is_constant(self):
        return self.body.is_constant()

    def count_constants(self):
        return self.body.count_constants()

    def derive_with_constants(self, constants):
        return New(self.body.derive_with_constants(constants), self.type, self.probability)
