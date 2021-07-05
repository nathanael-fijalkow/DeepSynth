'''
Objective: define a type system.
A type can be either PolymorphicType, PrimitiveType, Arrow, or List
'''

# make sure hash is deterministic
PYTHONHASHSEED = 0
class Type:
    '''
    Object that represents a type
    '''
    def __eq__(self, other):
        '''
        Type equality
        '''
        b = isinstance(self, Type) and isinstance(other, Type)
        b2 = isinstance(self,UnknownType) and isinstance(other,UnknownType)
        b2 = b2 or (isinstance(self,PolymorphicType) and isinstance(other,PolymorphicType) and self.name == other.name)
        b2 = b2 or (isinstance(self,PrimitiveType) and isinstance(other,PrimitiveType) and self.type == other.type)
        b2 = b2 or (isinstance(self,Arrow) and isinstance(other,Arrow) and self.type_in.__eq__(other.type_in) and self.type_out.__eq__(other.type_out))
        b2 = b2 or (isinstance(self,List) and isinstance(other,List) and self.type_elt.__eq__(other.type_elt))
        return b and b2

    def __gt__(self, other): True
    def __lt__(self, other): False
    def __ge__(self, other): True
    def __le__(self, other): False

    def __hash__(self):
        return self.hash

    def returns(self):
        if isinstance(self,Arrow):
            return self.type_out.returns()
        else:
            return self

    def arguments(self):
        if isinstance(self,Arrow):
            return [self.type_in] + self.type_out.arguments()
        else:
            return []

    def ends_with(self, other):
        '''
        Checks whether other is a suffix of self and returns the list of arguments

        Example: 
        self = Arrow(INT, Arrow(INT, INT))
        other = Arrow(INT, INT)
        ends_with(self, other) = [INT]

        self = Arrow(Arrow(INT, INT), Arrow(INT, INT))
        other = INT
        ends_with(self, other) = [Arrow(INT, INT), INT]
        '''
        return self.ends_with_rec(other, [])

    def ends_with_rec(self, other, arguments_list):
        if self == other:
            return arguments_list
        if isinstance(self, Arrow):
            arguments_list.append(self.type_in)
            return self.type_out.ends_with_rec(other, arguments_list)
        return None

    def size(self):
        if isinstance(self,(PrimitiveType,PolymorphicType)):
            return 1
        if isinstance(self,Arrow):
            return self.type_in.size() + self.type_out.size()
        if isinstance(self,List) and isinstance(self.type_elt,(PrimitiveType,PolymorphicType)):
            return 2
        if isinstance(self,List) and isinstance(self.type_elt,List) \
        and isinstance(self.type_elt.type_elt,(PrimitiveType,PolymorphicType)):
            return 3
        # We do not want List(List(List(...)))
        return 100

    def find_polymorphic_types(self):
        set_types = set()
        return self.find_polymorphic_types_rec(set_types)

    def find_polymorphic_types_rec(self, set_types):
        if isinstance(self,PolymorphicType):
            if not self.name in set_types:
                set_types.add(self.name)
        if isinstance(self,Arrow):
            set_types = self.type_in.find_polymorphic_types_rec(set_types)
            set_types = self.type_out.find_polymorphic_types_rec(set_types)
        if isinstance(self,List):
            set_types = self.type_elt.find_polymorphic_types_rec(set_types)
        return set_types

    def decompose_type(self):
        '''
        Finds the set of basic types and polymorphic types 
        '''
        set_basic_types = set()
        set_polymorphic_types = set()
        return self.decompose_type_rec(set_basic_types,set_polymorphic_types)

    def decompose_type_rec(self,set_basic_types,set_polymorphic_types):
        if isinstance(self,PrimitiveType):
            set_basic_types.add(self)
        if isinstance(self,PolymorphicType):
            set_polymorphic_types.add(self)
        if isinstance(self,Arrow):
            self.type_in.decompose_type_rec(set_basic_types,set_polymorphic_types)
            self.type_out.decompose_type_rec(set_basic_types,set_polymorphic_types)
        if isinstance(self,List):
            self.type_elt.decompose_type_rec(set_basic_types,set_polymorphic_types)
        return set_basic_types,set_polymorphic_types

    def unify(self, other):
        '''
        Checks whether self can be instantiated into other
        # and returns the least unifier as a dictionary {t : type}
        # mapping polymorphic types to types.

        IMPORTANT: We assume that other does not contain polymorphic types.

        Example: 
        * list(t0) can be instantiated into list(int) and the unifier is {t0 : int}
        * list(t0) -> list(t1) can be instantiated into list(int) -> list(bool) 
        and the unifier is {t0 : int, t1 : bool}
        * list(t0) -> list(t0) cannot be instantiated into list(int) -> list(bool) 
        '''
        dic = {}
        if self.unify_rec(other, dic):
            return True
        else:
            return False

    def unify_rec(self, other, dic):
        if isinstance(self,PolymorphicType):
            if self.name in dic:
                return dic[self.name] == other
            else:
                dic[self.name] = other
                return True
        if isinstance(self,PrimitiveType):
            return isinstance(other,PrimitiveType) and self.type == other.type
        if isinstance(self,Arrow):
            return isinstance(other,Arrow) and self.type_in.unify_rec(other.type_in, dic) and self.type_out.unify_rec(other.type_out, dic)
        if isinstance(self,List):
            return isinstance(other,List) and self.type_elt.unify_rec(other.type_elt, dic)

    def apply_unifier(self, dic):
        if isinstance(self,PolymorphicType):
            if self.name in dic:
                return dic[self.name]
            else:
                return self
        if isinstance(self,PrimitiveType):
            return self
        if isinstance(self,Arrow):
            new_type_in = self.type_in.apply_unifier(dic)
            new_type_out = self.type_out.apply_unifier(dic)
            return Arrow(new_type_in, new_type_out)
        if isinstance(self,List):
            new_type_elt = self.type_elt.apply_unifier(dic)
            return List(new_type_elt)

class PolymorphicType(Type):
    def __init__(self, name):
        assert(isinstance(name,str))
        self.name = name
        self.hash = hash(name)

    def __repr__(self):
        return str(self.name)

class PrimitiveType(Type):
    def __init__(self, type_):
        assert(isinstance(type_,str))
        self.type = type_
        self.hash = hash(type_)


    def __repr__(self):
        return str(self.type)

class Arrow(Type):
    def __init__(self, type_in, type_out):
        assert(isinstance(type_in,Type))
        assert(isinstance(type_out,Type))
        self.type_in = type_in
        self.type_out = type_out
        self.hash = hash((type_in.hash,type_out.hash))
        # self.hash = hash(str(self))


    def __repr__(self):
        rep_in = repr(self.type_in)
        rep_out = repr(self.type_out)
        return "({} -> {})".format(rep_in, rep_out)

class List(Type):
    def __init__(self, _type):
        assert(isinstance(_type,Type))
        self.type_elt = _type
        self.hash = hash(18923 + _type.hash)


    def __repr__(self):
        if isinstance(self.type_elt,Arrow):
            return "list{}".format(self.type_elt)
        else:
            return "list({})".format(self.type_elt)

class UnknownType(Type):
    '''
    In case we need to define an unknown type
    '''
    def __init__(self):
        self.type = ""
        self.hash = 1984

    def __repr__(self):
        return "UnknownType"

INT = PrimitiveType('int')
BOOL = PrimitiveType('bool')
STRING = PrimitiveType('str')