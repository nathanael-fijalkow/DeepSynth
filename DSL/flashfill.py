from type_system import *
import re

primitive_types = {
  '++' : Arrow(STRING,Arrow(STRING,STRING)),
  'replace' : Arrow(STRING,Arrow(STRING,Arrow(STRING,STRING))),
  'at' : Arrow(STRING,Arrow(INT,STRING)),
  'into2str' : Arrow(INT,STRING),
  'str.ite' : Arrow(BOOL,Arrow(STRING,Arrow(STRING,STRING))),
  'substr' : Arrow(STRING,Arrow(INT,Arrow(INT,STRING))),

  '+' : Arrow(INT,Arrow(INT,INT)),
  '-' : Arrow(INT,Arrow(INT,INT)),
  'len' : Arrow(STRING,INT),
  'str2int' : Arrow(STRING,INT),
  'int.ite' : Arrow(BOOL,Arrow(INT,Arrow(INT,INT))),
  'indexof' : Arrow(STRING, Arrow(STRING, INT)),

  '=' : Arrow(INT,Arrow(INT,BOOL)),
  'prefixof' : Arrow(STRING,Arrow(STRING,BOOL)),
  'suffixof' : Arrow(STRING,Arrow(STRING,BOOL)),
  'contains' : Arrow(STRING,Arrow(STRING,BOOL)),
  'constant' : STRING
}

def indexof(string1, string2) -> int:
  match = re.search(string1, string2)
  if match:
    match.start()
  else:
    -1

semantics = {
  '++' : lambda string1, string2: string1 + string2,
  'replace'  : lambda string1, string2, string3: re.sub(string1, string2, string3),
  'at' : lambda string1, int1: string1[int1] ,
  'int2str' : lambda int1: str(int1),
  'str.ite'  : lambda bool1, string1, string2: string1 if bool1 else string2,
  'substr' : lambda string1, int1, int2: string1[int1:int2],
  '+' : lambda int1, int2: int1 + int2,
  '-'  : lambda int1, int2: int1 - int2,
  'len' : lambda string1: len(string1),
  'str2int' : lambda string1: int(string1),
  'int.ite'  : lambda bool1, int1, int2: int1 if bool1 else int2,
  'indexof' : lambda string1: lambda string2: indexof(string1, string2),
  'prefixof' : lambda string1, string2: string2.startswith(string1),
  'suffixof'  : lambda string1, string2: string2.endswith(string1),
  'contains' : lambda string1, string2: string1 in string2,
  'constant' : None
}

no_repetitions = {}