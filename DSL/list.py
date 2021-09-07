from type_system import *
from program import *

import math
from functools import reduce

t0 = PolymorphicType('t0')
t1 = PolymorphicType('t1')

def _flatten(l): return [x for xs in l for x in xs]

def _range(n):
    if n < 100: return list(range(n))
    raise ValueError()

def _if(c): return lambda t: lambda f: t if c else f

def _and(x): return lambda y: x and y


def _or(x): return lambda y: x or y


def _addition(x): return lambda y: x + y


def _subtraction(x): return lambda y: x - y


def _multiplication(x): return lambda y: x * y


def _negate(x): return -x


def _reverse(x): return list(reversed(x))


def _append(x): return lambda y: x + y


def _cons(x): return lambda y: [x] + y


def _car(x): return x[0]


def _cdr(x): return x[1:]


def _isEmpty(x): return x == []


def _single(x): return [x]


def _slice(x): return lambda y: lambda l: l[x:y]


def _map(f): return lambda l: list(map(f, l))


def _zip(a): return lambda b: lambda f: list(map(lambda x,y: f(x)(y), a, b))


def _mapi(f): return lambda l: list(map(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l)))


def _reduce(f): return lambda x0: lambda l: reduce(lambda a, x: f(a)(x), l, x0)


def _reducei(f): return lambda x0: lambda l: reduce(
    lambda a, t: f(t[0])(a)(t[1]), enumerate(l), x0)


def _fold(l): return lambda x0: lambda f: reduce(
    lambda a, x: f(x)(a), l[::-1], x0)


def _eq(x): return lambda y: x == y


def _eq0(x): return x == 0


def _a1(x): return x + 1


def _d1(x): return x - 1


def _mod(x): return lambda y: y % x if x != 0 else None


def _not(x): return not x


def _gt(x): return lambda y: x > y


def _index(j): return lambda l: l[j]


def _replace(f): return lambda lnew: lambda lin: _flatten(
    lnew if f(i)(x) else [x] for i, x in enumerate(lin))


def _isPrime(n):
    return n in {
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199}


def _isSquare(n):
    return int(math.sqrt(n)) ** 2 == n


def _appendmap(f): lambda xs: [y for x in xs for y in f(x)]


def _filter(f): return lambda l: list(filter(f, l))


def _any(f): return lambda l: any(f(x) for x in l)


def _all(f): return lambda l: all(f(x) for x in l)


def _find(x):
    def _inner(l):
        try:
            return l.index(x)
        except ValueError:
            return -1
    return _inner

def _unfold(x): return lambda p: lambda f: lambda n: __unfold(p, f, n, x)

def __unfold(p, f, n, x, recursion_limit=20):
    if recursion_limit <= 0:
        raise ValueError
    if p(x):
        return []
    return [f(x)] + __unfold(p, f, n, n(x), recursion_limit - 1)

def _fix(argument):
    def inner(body):
        recursion_limit = [20]

        def fix(x):
            def r(z):
                recursion_limit[0] -= 1
                if recursion_limit[0] <= 0:
                    raise RecursionDepthExceeded()
                else:
                    return fix(z)

            return body(r)(x)
        return fix(argument)

    return inner


def curry(f): return lambda x: lambda y: f((x, y))


def _fix2(a1):
    return lambda a2: lambda body: \
        _fix((a1, a2))(lambda r: lambda n_l: body(curry(r))(n_l[0])(n_l[1]))

def _match(l):
    return lambda b: lambda f: b if l == [] else f(l[0])(l[1:])


def _miter(k, f, x):
    if k <= 0:
        return x
    return _miter(k-1, f, f(x))
semantics = {
        "empty" : [],
        "cons" : _cons,
        "car" : _car,
        "cdr" : _cdr,
        "empty?" : _isEmpty,
        "gt?": _gt,
        "le?": lambda x: lambda y: x <= y,
        "not": lambda x: not x,
        "max": lambda x: lambda y: max(x, y),
        "min": lambda x: lambda y: min(x, y),
        "if" : _if,
        "eq?" : _eq,
        "*" : _multiplication,
        "+" : _addition,
        "-" : _subtraction,
        "length" : len,
        "0" : 0,
        "1" : 1,
        "2" : 2,
        "3" : 3,
        "4" : 4,
        "5" : 5,
        "append": lambda x: lambda l: l + [x],
        "range" : _range,
        "map" : _map,
        "unfold" : _unfold,
        "index" : _index,
        "fold" : _fold,
        "is-mod": lambda x: lambda y: y % x == 0 if x != 0 else False,
        "mod": _mod,
        "iter": _miter,
        "is-prime" : _isPrime,
        "is-square" : _isSquare,
        "filter": lambda f: lambda l: [x for x in l if f(x)]
        }

primitive_types = {
        "empty": List(t0),
        "cons": Arrow(t0,Arrow(List(t0),List(t0))),
        "car": Arrow(List(t0), t0),
        "cdr": Arrow(List(t0), List(t0)),
        "empty?": Arrow(List(t0), BOOL),
        "max": Arrow(INT, Arrow(INT, INT)),
        "min": Arrow(INT, Arrow(INT, INT)),
        "gt?": Arrow(INT, Arrow(INT, BOOL)),
        "le?": Arrow(INT, Arrow(INT, BOOL)),
        "not": Arrow(BOOL, BOOL),
        "if": Arrow(BOOL, Arrow(t0, Arrow(t0, t0))),
        "eq?": Arrow(INT, Arrow(INT, BOOL)),
        "*": Arrow(INT, Arrow(INT, INT)),
        "+": Arrow(INT, Arrow(INT, INT)),
        "-": Arrow(INT, Arrow(INT, INT)),
        "length": Arrow(List(t0), INT),
        "0": INT,
        "1": INT,
        "2": INT,
        "3": INT,
        "4": INT,
        "5": INT,
        "range": Arrow(INT, List(INT)),
        "map": Arrow(Arrow(t0, t1), Arrow(List(t0), List(t1))),
        "iter": Arrow(INT, Arrow(Arrow(t0, t0), Arrow(t0, t0))),
        "append": Arrow(t0, Arrow(List(t0), List(t0))),
        "unfold": Arrow(t0, Arrow(Arrow(t0,BOOL), Arrow(Arrow(t0,t1), Arrow(Arrow(t0,t0), List(t1))))),
        "index": Arrow(INT, Arrow(List(t0), t0)),
        "fold": Arrow(List(t0), Arrow(t1, Arrow(Arrow(t0, Arrow(t1, t1)), t1))),
        "is-mod": Arrow(INT, Arrow(INT, BOOL)),
        "mod": Arrow(INT, Arrow(INT, INT)),
        "is-prime": Arrow(INT, BOOL),
        "is-square": Arrow(INT, BOOL),
        "filter": Arrow(Arrow(t0, BOOL), Arrow(List(t0), List(t0))),
        }

# (unfold (* (mod 5 0) (index 4 var0)) false)
# fold: 'a list -> ('b -> 'a -> 'b) -> 'b -> 'b
# (range (fold (cdr var0) -2 pow))
