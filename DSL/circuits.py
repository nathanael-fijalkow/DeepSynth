from type_system import *
 
primitive_types = {
  'and' : Arrow(BOOL,Arrow(BOOL,BOOL)),
  'or'  : Arrow(BOOL,Arrow(BOOL,BOOL)),
  'xor' : Arrow(BOOL,Arrow(BOOL,BOOL)),
  'not' : Arrow(BOOL,BOOL),
  }

semantics = {
	'and' : lambda bool1: lambda bool2: bool1 and bool2,
	'or'  : lambda bool1: lambda bool2: bool1 or bool2,
	'xor' : lambda bool1: lambda bool2: bool1^bool2,
	'not' : lambda bool: not bool
}

no_repetitions = {}
