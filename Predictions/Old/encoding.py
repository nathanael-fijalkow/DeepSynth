import torch

from program import Program, Function, Variable, BasicPrimitive, New

class Encoding():
    '''
    Objet that can embed inputs, outputs and programs
    for simplicity we only embed arguments of type list[int] here (there is another file with a more generic implementation with the possibility to work with any type)

    cfg: a cfg template

    size_max: size max of an input or an output (= length of the associated list)

    nb_inputs_max: maximum number of inputs

    example; if inputs/output = [[input_1, input_2, etc..], output] = [[[11,20],[3]], [12,2]] and size_max = 2, nb_inputs_max = 3 the encoding is [11,1,20,1,3,1,0,0,0,0,0,0, 12,1,2,1]
    '''

    def __init__(self, cfg, size_max, nb_inputs_max) -> None:
        self.cfg = cfg
        self.size_max = size_max
        self.nb_inputs_max = nb_inputs_max
        # self.io_dim: the dimension of the concatenation of an input/output pair
        self.io_dim = 2*size_max*(1+nb_inputs_max)
        self.output_dim = 0
        counter = 0
        # self.hash_table[(S,P)] ::= position of the transition (S,P) in the final layer of the neural network
        self.hash_table = {}
        for S in cfg.rules:
            self.output_dim += len(cfg.rules[S])
            for P in cfg.rules[S]:
                self.hash_table[(S, P)] = counter
                counter += 1

    def encode_program(self, program, S=None, tensor=None):
        '''
        take a program and output a tensor of dimension #(transitions) (it sets a 1 for the transitions used to derive the program and 0 otherwise)
        '''
        if S == None:
            S = self.cfg.start
        if tensor == None:
            tensor = torch.zeros(self.output_dim)
        if isinstance(program, Function):
            F = program.function
            args_P = program.arguments
            tensor[self.hash_table[(S, F)]] += 1
            for i, arg in enumerate(args_P):
                self.encode_program(
                    arg, self.cfg.rules[S][F][i], tensor)

        if isinstance(program, (BasicPrimitive, Variable)):
            tensor[self.hash_table[(S, program)]] += 1

        if S == self.cfg.start:
            return tensor

    def encode_single_arg(self, arg):
        '''
        embed a single list of floats (for example an input or an output)
        '''
        res = torch.zeros(2*self.size_max)
        for i, e in enumerate(arg):
            if i >= self.size_max:
                print("Oh oh, this has too many elements: ", arg)
                assert(False)  # if more elements than size_max, rise a problem
            res[2*i] = e
            # flag to say to the neural net that the previous value is a real one
            res[2*i+1] = 1
        return res

    def encode_IO(self, args):
        '''
        embed a list of inputs and its associated output
        args = list containing the inputs and the associated output in the format args ::= [[i1,i2,...],o], where any i1, i2, .. and o are lists of floats
        '''
        res = []
        inputs, output = args
        for i in range(self.nb_inputs_max):  # if more inputs there are ignored
            try:
                input = inputs[i]
                embedded_input = self.encode_single_arg(input)
                res.append(embedded_input)
            except:
                res.append(torch.zeros(2*self.size_max))

        res.append(self.encode_single_arg(output))
        return torch.cat(res)

    def encode_all_examples(self, IOs):
        '''
        Embed a list of IOs (it simply stacks the embedding of a single inputs/output pair)
        '''
        res = []
        for IO in IOs:
            res.append(self.encode_IO(IO))
        return torch.stack(res)
