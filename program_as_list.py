from program import BasicPrimitive, Function, Lambda, New, Variable

def evaluation_from_compressed(program_compressed, dsl, environment, target_type):
    stack = []
    (P, sub_program) = program_compressed

    while P:
        if isinstance(P, (New, BasicPrimitive)):
            try:
                list_arguments = P.type.ends_with(target_type)
                evaluated_arguments = []
                evaluation = P.eval_naive(dsl, environment)

                for _ in range(len(list_arguments)):
                    evaluated_arguments.append(stack.pop())
                evaluated_arguments.reverse()
                for evaluated_arg in evaluated_arguments:
                    evaluation = evaluation(evaluated_arg)
                stack.append(evaluation)
            except (IndexError, ValueError, TypeError):
                stack.append(None)

        elif isinstance(P, Variable):
            stack.append(P.eval_naive(dsl, environment))

        elif isinstance(P, Lambda):
            eval_lambda = P.eval_naive(dsl, environment)
            arg_lambda = stack.pop()
            stack.append(eval_lambda(arg_lambda))

        if sub_program:
            (P, sub_program) = sub_program[0], sub_program[1]
        else:
            P = None

    return stack.pop()

def reconstruct_from_compressed(program, target_type):
    program_as_list = []
    list_from_compressed(program, program_as_list)
    program_as_list.reverse()
    return reconstruct_from_list(program_as_list, target_type)

def list_from_compressed(program, program_as_list=None):
    (P, sub_program) = program
    if sub_program:
        list_from_compressed(sub_program, program_as_list)
    program_as_list.append(P)

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





# # USEFUL if a TOP DOWN approach is used in evaluation_from_compressed
# def evaluation_from_list(program_as_list, dsl, environment, target_type):
#     if len(program_as_list) == 1:
#         return program_as_list.pop().eval_naive(dsl, environment)
#     else:
#         P = program_as_list.pop()
#         if isinstance(P, (New, BasicPrimitive)):
#             try:
#                 list_arguments = P.type.ends_with(target_type)
#                 evaluated_arguments = [None] * len(list_arguments)
#                 for i in range(len(list_arguments)):
#                     evaluated_arguments[
#                         len(list_arguments) - i - 1
#                     ] = evaluation_from_list(
#                         program_as_list,
#                         dsl,
#                         environment,
#                         list_arguments[len(list_arguments) - i - 1],
#                     )
#                 evaluation = P.eval_naive(dsl, environment)
#                 for evaluated_arg in evaluated_arguments:
#                     evaluation = evaluation(evaluated_arg)
#                 return evaluation
#             except (IndexError, ValueError, TypeError):
#                 return None

#         if isinstance(P, Variable):
#             return P.eval_naive(dsl, environment)
#         assert False
