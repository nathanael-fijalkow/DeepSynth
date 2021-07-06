class ProgramAsList:
    def evaluation(self, dsl, environment):
        (P, sub_program) = self
        if sub_program: 
           pass 
        else:
           P.eval(stack)

P
[2,3]
environment = ([2,3], None)
evaluation(P, environment)

new_environment = (new_value, environment)

# (1, (var0, ((lambda (map (lambda ((+ var0) var1)))), None)))

TRANSLATE : obj vers int vers obj

TRANSLATE obj 4 =
Function(TRANSLATE, [obj, 4])

(TRANSLATE obj) 4 =
Function(Function(TRANSLATE,[obj]), [4])

obj 4 TRANSLATE
(4, (obj, TRANSLATE))

TO DO!
# Once we have a good JSON format for PCFG and we can test semantic_experiments
        
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