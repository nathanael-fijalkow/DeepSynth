import os
import sys


from type_system import *
from cons_list import index

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

    def typeless_eq(self, other, verbose=False):
        if verbose:
            print(
                "checking:\n    1:{}\nclass: {}\n    2:{}\nclass: {}".format(
                    self,
                    self.__class__.__name__,
                    other,
                    other.__class__.__name__,
                )
            )
        b = isinstance(self, Program) and isinstance(other, Program)
        b2 = (
            isinstance(self, Variable)
            and isinstance(other, Variable)
            and self.variable == other.variable
        )
        b2 = b2 or (
            isinstance(self, Function)
            and isinstance(other, Function)
            and self.function.typeless_eq(other.function, verbose)
            and len(self.arguments) == len(other.arguments)
            and all(
                [
                    x.typeless_eq(y, verbose)
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
            and (self.body).typeless_eq(other.body, verbose)
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

class Variable(Program):
    def __init__(self, variable, type_=UnknownType(), probability={}):
        # self.variable is a natural number
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
    def __init__(self, primitive, type_=UnknownType(), probability={}):
        # assert isinstance(primitive, str)
        self.primitive = primitive
        # assert isinstance(type_, Type)
        self.type = type_
        self.hash = hash(primitive) + self.type.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return format(self.primitive)

    def eval(self, dsl, environment, i):
        return dsl.semantics[self.primitive]

    def eval_naive(self, dsl, environment):
        return dsl.semantics[self.primitive]

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

def reconstruct_from_list(program_as_list, target_type):
    if len(program_as_list) == 1:
        return program_as_list.pop()
    else:
        P = program_as_list.pop()
        if isinstance(P, (New, BasicPrimitive)):
            list_arguments = P.type.ends_with(target_type)
            arguments = [None] * len(list_arguments)
            for i in range(len(list_arguments)):
                arguments[len(list_arguments) - i - 1] = reconstruct_from_list(
                    program_as_list, list_arguments[len(list_arguments) - i - 1]
                )
            return Function(P, arguments)
        if isinstance(P, Variable):
            return P
        assert False

def reconstruct_from_compressed(program, target_type):
    program_as_list = []
    list_from_compressed(program, program_as_list)
    program_as_list.reverse()
    return reconstruct_from_list(program_as_list, target_type)


def list_from_compressed(program, program_as_list=[]):
    (P, sub_program) = program
    if sub_program:
        list_from_compressed(sub_program, program_as_list)
    program_as_list.append(P)
