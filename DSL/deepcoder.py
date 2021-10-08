from type_system import *
from program import *

t0 = PolymorphicType('t0')
t1 = PolymorphicType('t1')

def access(i,l):
	if i is None: 
		return None 
	elif ((i>=0 and len(l)>i) or (i<0 and len(l)>=-i)): 
		return l[i] 
	else: 
		return None

def scanl(op):
	def aux(l):
		if len(l) == 0: return []
		else:
			y = [l[0]]
			for x in l[1:]:
				last = y[-1]
				y.append(op(last, x)) 
		return y
	return aux

semantics = {
	'HEAD': lambda l: l[0] if len(l)>0 else None,
	'TAIL': lambda l: l[-1] if len(l) > 0 else None,
	'ACCESS': lambda i: lambda l: access(i, l),
	'MINIMUM': lambda l: min(l) if len(l)>0 else None,
	'MAXIMUM': lambda l: max(l) if len(l)>0 else None,
	'LENGTH': lambda l: len(l),
	'COUNT[<0]': lambda l: len([x for x in l if x<0]),
	'COUNT[>0]': lambda l: len([x for x in l if x>0]),
	'COUNT[EVEN]': lambda l: len([x for x in l if x%2==0]),
	'COUNT[ODD]': lambda l: len([x for x in l if x%2==1]),
	'SUM': lambda l: sum(l),

	'TAKE': lambda i: lambda l: l[:i],
	'DROP': lambda i: lambda l: l[i:],
	'SORT': lambda l: sorted(l),
	'REVERSE': lambda l: l[::-1],
	'FILTER[<0]': lambda l: [x for x in l if x<0],
	'FILTER[>0]': lambda l: [x for x in l if x>0],
	'FILTER[EVEN]': lambda l: [x for x in l if x%2==0],
	'FILTER[ODD]': lambda l: [x for x in l if x % 2 == 1],
	'MAP[+1]': lambda l: [x + 1 for x in l],
	'MAP[-1]': lambda l: [x - 1 for x in l],
	'MAP[*2]': lambda l: [x * 2 for x in l],
	'MAP[/2]': lambda l: [int(x / 2) for x in l],
	'MAP[*3]': lambda l: [x * 3 for x in l],
	'MAP[/3]': lambda l: [int(x / 3) for x in l],
	'MAP[*4]': lambda l: [x * 4 for x in l],
	'MAP[/4]': lambda l: [int(x / 4) for x in l],
	'MAP[**2]': lambda l: [x ** 2 for x in l],
	'MAP[*-1]': lambda l: [-x for x in l],
	'ZIPWITH[+]': lambda l1: lambda l2: [x + y for (x,y) in zip(l1,l2)],
	'ZIPWITH[-]': lambda l1: lambda l2: [x - y for (x,y) in zip(l1,l2)],
	'ZIPWITH[*]': lambda l1: lambda l2: [x * y for (x,y) in zip(l1,l2)],
	'ZIPWITH[max]': lambda l1: lambda l2: [(x if x > y else y) for (x,y) in zip(l1,l2)],
	'ZIPWITH[min]': lambda l1: lambda l2: [(y if x > y else x) for (x,y) in zip(l1,l2)],
	'SCAN1L[+]': scanl(lambda x, y: x + y),
	'SCAN1L[-]': scanl(lambda x, y: x - y),
	'SCAN1L[*]': scanl(lambda x, y: x * y),
	'SCAN1L[min]': scanl(lambda x, y: min(x,y)),
	'SCAN1L[max]': scanl(lambda x, y: max(x,y)),

	# 'MAP': lambda f: lambda l: list(map(f, l)),
}

primitive_types = {
	'HEAD': Arrow(List(INT),INT),
	'TAIL': Arrow(List(INT), INT),
	'ACCESS': Arrow(List(INT),INT),
	'MINIMUM': Arrow(List(INT),INT),
	'MAXIMUM': Arrow(List(INT),INT),
	'LENGTH': Arrow(List(INT),INT),
	'COUNT[<0]': Arrow(List(INT),INT),
	'COUNT[>0]': Arrow(List(INT),INT),
	'COUNT[EVEN]': Arrow(List(INT),INT),
	'COUNT[ODD]': Arrow(List(INT),INT),
	'SUM': Arrow(List(INT),INT),
	'TAKE': Arrow(INT, Arrow(List(INT), List(INT))),
	'DROP': Arrow(INT, Arrow(List(INT), List(INT))),
	'SORT': Arrow(List(INT), List(INT)),
	'REVERSE': Arrow(List(INT), List(INT)),
	'FILTER[<0]': Arrow(List(INT), List(INT)),
	'FILTER[>0]': Arrow(List(INT), List(INT)),
	'FILTER[EVEN]': Arrow(List(INT), List(INT)),
	'FILTER[ODD]': Arrow(List(INT), List(INT)),
	'MAP[+1]': Arrow(List(INT), List(INT)),
	'MAP[-1]': Arrow(List(INT), List(INT)),
	'MAP[*2]': Arrow(List(INT), List(INT)),
	'MAP[/2]': Arrow(List(INT), List(INT)),
	'MAP[*-1]': Arrow(List(INT), List(INT)),
	'MAP[**2]': Arrow(List(INT), List(INT)),
	'MAP[*3]': Arrow(List(INT), List(INT)),
	'MAP[/3]': Arrow(List(INT), List(INT)),
	'MAP[*4]': Arrow(List(INT), List(INT)),
	'MAP[/4]': Arrow(List(INT), List(INT)),
	'ZIPWITH[+]': Arrow(List(INT), Arrow(List(INT), List(INT))),
	'ZIPWITH[-]': Arrow(List(INT), Arrow(List(INT), List(INT))),
	'ZIPWITH[*]': Arrow(List(INT), Arrow(List(INT), List(INT))),
	'ZIPWITH[min]': Arrow(List(INT), Arrow(List(INT), List(INT))),
	'ZIPWITH[max]': Arrow(List(INT), Arrow(List(INT), List(INT))),
	'SCAN1L[+]': Arrow(List(INT), List(INT)),
	'SCAN1L[-]': Arrow(List(INT), List(INT)),
	'SCAN1L[*]': Arrow(List(INT), List(INT)),
	'SCAN1L[min]': Arrow(List(INT), List(INT)),
	'SCAN1L[max]': Arrow(List(INT), List(INT)),
	# 'MAP': Arrow(Arrow(t0,t1),Arrow(List(t0),List(t1))),
	}

no_repetitions = {
	'SORT',
	'REVERSE',
	'FILTER[<0]',
	'FILTER[>0]',
	'FILTER[EVEN]',
	'FILTER[ODD]',
	}
