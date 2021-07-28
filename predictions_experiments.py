import logging

from torch._C import dtype

from dsl import DSL
from program import Program, Function, Variable, BasicPrimitive, New
from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
import torch
from torch import nn
from pcfg import PCFG

device = 'cpu'
# A block is a concatenation of a linear layer + a sigmoid
# This is a "map_style" Dataset, it is also possible to define a Dataset with an iterator instead of a __getitem__ function (useful for streaming/random inputs/output pair) 
class Data(torch.utils.data.Dataset):
    def __init__(self, tasks, programs, transform=None):
        self.tasks = tasks
        self.programs = programs
        self.transform = transform

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        x = self.tasks[index]
        y = self.programs[index]
        if self.transform:
            x = self.transform.embed_all_examples(x)
            y = self.transform.embed_program(y)
        return x, y



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

    def __init__(self, template_cfg, embedder, size_hidden, min_int = 0, max_int = 10):
        super(Net, self).__init__()

        self.template_cfg = template_cfg
        self.embedder = embedder
        self.min_int = min_int
        self.max_int = max_int
        self.size_lexicon = self.max_int - self.min_int + 1
        self.dim_embedding = 2

        self.io_dim = embedder.io_dim
        self.output_dim = embedder.output_dim

        self.embed = nn.Embedding(self.size_lexicon, self.dim_embedding)
        # hidden layers
        self.hidden = nn.Sequential(
            block(self.io_dim*self.dim_embedding, size_hidden),
            block(size_hidden, size_hidden),
            block(size_hidden, size_hidden),
        )

        # final activation
        self.final_layer = nn.Sequential(
            nn.Linear(size_hidden, self.output_dim),
            nn.Sigmoid()
        )

    # data = list d'encoding de IOs
    # renvoyer un tenseur
    # colate qui crée une liste de IOs/ tenseur de programmes
    def forward(self, list_IOs):
        '''
        Function for completing a forward pass of the Net, and output the array of transition probabilities
        Parameters:
            data: a tensor with dimensions (batch_size, io_nb, io_dim) (in pytorch, this is named (N,*,H_in))
        '''
        res = []
        for x in list_IOs:
            x = x.long()
            x = x - self.min_int # translate to have everything between 0 and size_lexicon
            x = self.embed(x)
            x = torch.flatten(x,start_dim=1)
            x = self.hidden(x)
        # print(x.shape)
        # Average along any column for any 2d matrix in the batch
            x = torch.mean(x, -2)
            x = self.final_layer(x)  # get the predictions
            res.append(x)
        return torch.stack(res)

    def forward_grammar(self, list_IOs):
        '''
        Perform a forward pass and reconstruct the grammar
        '''
        res = []
        for x in list_IOs:
            x = x.long()
            x = x - self.min_int # translate to have everything between 0 and size_lexicon
            x = self.embed(x)
            x = torch.flatten(x,start_dim=1)
            x = self.hidden(x)
        # print(x.shape)
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
            res.append(PCFG(template_cfg.start, rules, max_program_depth=template_cfg.max_program_depth))

        # ?TODO? maybe normalize the grammar before outputing it??
        # return grammars
        return res

    def train(self, data, epochs=200):
        optimizer = torch.optim.Adam(self.parameters())

        for step in range(epochs):
            for data in trainset:  # batch of data
                X, y = data
                optimizer.zero_grad()
                output = self(X)
                loss_value = loss(output, y)
                loss_value.backward()
                optimizer.step()

            if step % 100 == 0:
                logging.debug("optimization step {}\tbinary cross entropy {}".format(
                    step, float(loss_value)))

    def test(self, programs, tasks):
        for task, program in zip(tasks, programs):
            grammar = self.forward_grammar(
                self.embedder.embed_all_examples(task))
            # grammar = grammar.normalise()
            # program = self.embedder.embed_program(program)
            # print("predicted grammar {}".format(grammar))
            print("intended program {}\nprobability {}".format(
                program, grammar.probability_program(self.template_cfg.start, program)))



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


#E = Encoding(template_cfg, 5, 5)
# I must be a list of list of floats (all inputs)
# 0 is a list of floats (the output)
# IO is a "pair" inputs/output as IO = [I,O], this is what we can used to feed the embedded as E.embed_IO(IO)
# TOY EXAMPLE
# I = [[10]]
# O = [1]
# I2 = [[77, 100], [33, 66]]
# O2 = [0]
# IO = [I, O]  # a single I/O example
# IO2 = [I2, O2]
# IOs = [IO, IO2]  # several I/O examples.
# print(E.embed_IO(IO))
# x = E.embed_all_examples(IOs)
# print(x)
# NN = Net(template_cfg, E, 10)  # a model with hidden layers of size 10
# print(NN(x))  # a forward pass: return the array of transition probabilities
# # a forward pass + the reconstruction of the grammar
# # print(NN.forward_grammar(x))


# --------- LEARNING, a toy example

# torch.device(device)

# # path to use a saved model
# # PATH_IN = "saved_models/test" #path for loading a model saved externally

# # path to save the model after the training
# # PATH_OUT = "saved_models/test_" + str(datetime.datetime.now())

# EPOCHS = 1_000

# # Loss
# loss = torch.nn.BCELoss(reduction='mean')

# # Models
# model = Net(template_cfg, E, 10)
# trainset = [(model.embedder.embed_all_examples([IO, IO]), model.embedder.embed_program(programs[0])),
#             (model.embedder.embed_all_examples([IO2]), model.embedder.embed_program(programs[1]))]
# # to use a saved model
# # M = torch.load(PATH_IN)

# # Optimizers
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# # optimizer = torch.optim.SGD(M.parameters(), lr=0.01, momentum=0.9)

# for epoch in range(EPOCHS):
#     for data in trainset:  # batch of data
#         X, y = data
#         model.zero_grad()
#         output = model(X)
#         loss_value = loss(output, y)
#         if epoch % 100 == 0:
#             print("optimization step", epoch,
#                   "\tbinary cross entropy ", float(loss_value))
#         loss_value.backward()
#         optimizer.step()

# print('\ntheoretical transitions:\n', E.embed_program(programs[0]))
# print('\n\npredicted transitions:\n', model(E.embed_all_examples([IO])))
# print('\n\ndifferences:\n', E.embed_program(
#     programs[0])-model(E.embed_all_examples([IO])))
# print('\n\nAssociated grammar:\n',
#       model.forward_grammar(E.embed_all_examples([IO])))
# G = model.forward_grammar(E.embed_all_examples([IO]))
# tasks = [[IO], [IO2]]
# model.test(programs, tasks)
# torch.save(model, PATH_OUT)

# training = Data(tasks, programs, E)
# X, y = training.__getitem__(0)
# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(training, batch_size=2, shuffle=True)

# for train_features, train_labels in train_dataloader: #it comes in batch
#     print(train_features, train_labels)