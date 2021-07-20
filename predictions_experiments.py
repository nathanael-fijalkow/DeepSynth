from dsl import DSL
from program import Program, Function, Variable, BasicPrimitive, New
from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
import torch
from torch import nn

device = 'cpu'

# A block is a concatenation of a linear layer + a sigmoid


def block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Sigmoid()
    )


class Net(nn.Module):
    '''
    Predictor Class
    Args:
        template_cfg: a cfg template
        embedder: a objet that can embed inputs, ouputs and programs
        size_hidden: size of a hidden layer
        output_dim: dimension of the output predictions (= number of transitions in the PCFG)        
    '''

    def __init__(self, template_cfg, embedder, size_hidden):
        super(Net, self).__init__()

        self.template_cfg = template_cfg
        self.embedder = embedder
        self.io_dim = embedder.io_dim
        self.output_dim = embedder.output_dim

        # hidden layers
        self.hidden = nn.Sequential(
            block(self.io_dim, size_hidden),
            block(size_hidden, size_hidden),
            block(size_hidden, size_hidden),
        )

        # final activation
        self.final_layer = nn.Sequential(
            nn.Linear(size_hidden, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        '''
        Function for completing a forward pass of the Net, and output the array of transition probabilities
        Parameters:
            data: a tensor with dimensions (batch_size, io_nb, io_dim) (in pytorch, this is named (N,*,H_in))
        '''
        x = self.hidden(data)
        # Average along any column for any 2d matrix in the batch
        x = torch.mean(x, -2)
        x = self.final_layer(x)  # get the predictions
        return x

    def forward_grammar(self, data):
        '''
        Perform a forward pass and reconstruct the grammar
        '''
        x = self.hidden(data)
        # Average along any column for any 2d matrix in the batch
        x = torch.mean(x, -2)
        x = self.final_layer(x)  # get the predictions

        # reconstruct the grammar
        grammars = []
        rules = {}
        for S in self.template_cfg.rules:
            rules[S] = {}
            for P in self.template_cfg.rules[S]:
                # x[self.embedder.hash_table[(S,P)] is the predicted proba of the rule S -> P
                rules[S][P] = self.template_cfg.rules[S][P], float(
                    x[self.embedder.hash_table[(S, P)]])
        grammars.append(rules)

        # ?TODO? maybe normalize the grammar before outputing it??
        return grammars


class Embedding():
    '''
    Objet that can embed inputs, ouputs and programs
    for simplicity we only embed arguments of type list[float] here (there is another file with a more generic implementation with the possibility to work with any type)

    template_cfg: a cfg template
    size_max: size max of an input or an output (= length of the associated list)
    nb_inputs_max: maximum number of inputs

    self.io_dim: the dimension of the concatenation of an input/output pair

    example; if inputs/output = [[input_1, input_2, etc..], output] = [[[11,20],[3]], [12,2]] and size_max = 2, nb_inputs_max = 3 the encoding is [11,1,20,1,3,1,0,0,0,0,0,0, 12,1,2,1]
    '''

    def __init__(self, template_cfg, size_max, nb_inputs_max) -> None:
        self.template_cfg = template_cfg
        self.size_max = size_max
        self.nb_inputs_max = nb_inputs_max
        self.io_dim = 2*size_max*(1+nb_inputs_max)
        self.output_dim = 0
        counter = 0
        self.hash_table = {} # self.hash_table[(S,P)] ::= position of the transition (S,P) in the final layer of the neural network
        for S in template_cfg.rules:
            self.output_dim += len(template_cfg.rules[S])
            for P in template_cfg.rules[S]:
                self.hash_table[(S, P)] = counter
                counter += 1

    def embed_program(self, program, S=None, tensor=None):
        '''
        take a program and output a tensor of dimension #(transitions) (it sets a 1 for the transitions used to derive the program and 0 otherwise)
        '''
        if S == None:
            S = self.template_cfg.start
        if tensor == None:
            tensor = torch.zeros(self.output_dim)
        if isinstance(program, Function):
            F = program.function
            args_P = program.arguments
            tensor[self.hash_table[(S, F)]] += 1
            for i, arg in enumerate(args_P):
                self.embed_program(
                    arg, self.template_cfg.rules[S][F][0], tensor)

        if isinstance(program, (BasicPrimitive, Variable)):
            tensor[self.hash_table[(S, program)]] += 1

        if S == self.template_cfg.start:
            return tensor

    def embed_single_arg(self, arg):
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

    def embed_IO(self, args):
        '''
        embed a list of inputs and its associated output
        args = list containing the inputs and the associated output in the format args ::= [[i1,i2,...],o], where any i1, i2, .. and o are lists of floats
        '''
        res = []
        inputs, output = args
        for i in range(self.nb_inputs_max):  # if more inputs there are ignored
            try:
                input = inputs[i]
                embedded_input = self.embed_single_arg(input)
                res.append(embedded_input)
            except:
                res.append(torch.zeros(2*self.size_max))

        res.append(self.embed_single_arg(output))
        return torch.cat(res)

    def embed_all_examples(self, IOs):
        '''
        Embed a list of IOs (it simply stacks the embedding of a single inputs/output pair)
        '''
        res = []
        for IO in IOs:
            res.append(self.embed_IO(IO))
        return torch.stack(res)


# Example of use

primitive_types = {
    "if": Arrow(BOOL, Arrow(INT, INT)),
    "+": Arrow(INT, Arrow(INT, INT)),
    "0": INT,
    "1": INT,
    "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
    "lt": Arrow(INT, Arrow(INT, BOOL)),
}

semantics = {
    "if": lambda b: lambda x: lambda y: x if b else y,
    "+": lambda x: lambda y: x + y,
    "0": 0,
    "1": 1,
    "and": lambda b1: lambda b2: b1 and b2,
    "lt": lambda x: lambda y: x <= y,
}

programs = [
    Function(BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))),
             [BasicPrimitive("0", INT), BasicPrimitive("1", INT)], INT),
    BasicPrimitive("0", INT),
    BasicPrimitive("1", INT)
]

dsl = DSL(semantics, primitive_types)
type_request = Arrow(INT, INT)
template_cfg = dsl.DSL_to_CFG(type_request)


E = Embedding(template_cfg, 5, 5)
# I must be a list of list of floats (all inputs)
# 0 is a list of floats (the output)
# IO is a "pair" inputs/output as IO = [I,O], this is what we can used to feed the embedded as E.embed_IO(IO)
# TOY EXAMPLE
I = [[10]]
O = [1]
I2 = [[77, 100], [33, 66]]
O2 = [0]
IO = [I, O]  # a single I/O example
IO2 = [I2, O2]
IOs = [IO, IO2]  # several I/O examples.
print(E.embed_IO(IO))
x = E.embed_all_examples(IOs)
print(x)
NN = Net(template_cfg, E, 10)  # a model with hidden layers of size 10
print(NN(x))  # a forward pass: return the array of transition probabilities
# a forward pass + the reconstruction of the grammar
print(NN.forward_grammar(x))


# --------- LEARNING, a toy example

torch.device(device)

# path to use a saved model
# PATH_IN = "saved_models/test" #path for loading a model saved externally

# path to save the model after the training
# PATH_OUT = "saved_models/test_" + str(datetime.datetime.now())

EPOCHS = 3000

# Loss
loss = torch.nn.BCELoss(reduction='mean')

# Models
model = Net(template_cfg, E, 10)
trainset = [(model.embedder.embed_all_examples([IO,IO]), model.embedder.embed_program(programs[0])),
            (model.embedder.embed_all_examples([IO2]), model.embedder.embed_program(programs[1]))]
# to use a saved model
# M = torch.load(PATH_IN)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.SGD(M.parameters(), lr=0.01, momentum=0.9)

for epoch in range(EPOCHS):
    for data in trainset:  # batch of data
        X, y = data
        model.zero_grad()
        output = model(X)
        loss_value = loss(output, y)
        if epoch % 100 == 0:
            print("optimization step", epoch,
                  "\tbinary cross entropy ", float(loss_value))
        loss_value.backward()
        optimizer.step()

print('\ntheoretical transitions:\n',E.embed_program(programs[0])) 
print('\n\npredicted transitions:\n', model(E.embed_all_examples([IO])))
print('\n\nAssociated grammar:\n', model.forward_grammar(E.embed_all_examples([IO])))

# torch.save(model, PATH_OUT)
