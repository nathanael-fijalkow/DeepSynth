import logging

from torch._C import dtype

from dsl import DSL
from program import Program, Function, Variable, BasicPrimitive, New
from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
import torch
from torch import nn
from pcfg import PCFG

device = 'cpu'

def block(input_dim, output_dimension):
    return nn.Sequential(
        nn.Linear(input_dim, output_dimension),
        nn.Sigmoid()
    )

class GlobalRulesPredictor(nn.Module):
    '''
    cfg: a cfg template
    IOEncoder: encode inputs and outputs
    IOEmbedder: embeds inputs and outputs
    size_hidden: size for hidden layers
    '''
    def __init__(self, 
        cfg, 
        IOEncoder,
        IOEmbedder,
        size_hidden,
        ):
        super(GlobalRulesPredictor, self).__init__()

        self.cfg = cfg
        self.IOEncoder = IOEncoder
        self.IOEmbedder = IOEmbedder

        self.loss = torch.nn.BCELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.nb_epochs = 20

        self.init_RuleToIndex()

        # layers
        self.hidden = nn.Sequential(
            block(IOEncoder.output_dimension * self.IOEmbedder.output_dimension, size_hidden),
            # block(size_hidden, size_hidden),
            block(size_hidden, size_hidden),
        )
        # final layer
        self.final_layer = block(size_hidden, self.output_dimension)

    def init_RuleToIndex(self):
        self.output_dimension = 0

        index = 0
        # self.RuleToIndex[(S,P)] is the index position of the derivation (S,P)
        # in the final layer of the neural network
        self.RuleToIndex = {}
        for S in self.cfg.rules:
            self.output_dimension += len(self.cfg.rules[S])
            for P in self.cfg.rules[S]:
                self.RuleToIndex[(S, P)] = index
                index += 1

    def forward(self, batch_IOs):
        '''
        batch_IOs is a tensor of size
        (batch_size, IOEncoder.output_dimension, IOEmbedder.output_dimension) 
        '''
        res = []
        for x in batch_IOs:
            # print("size of x", x.size())
            x = self.IOEmbedder.forward_IOs(x)
            # print("size of x", x.size())
            x = torch.flatten(x, start_dim = 1)
            # print("size of x", x.size())
            x = self.hidden(x)
            # print("size of x", x.size())
            x = torch.mean(x, -2)
            # print("size of x", x.size())
            x = self.final_layer(x)
            # print("size of x", x.size())
            res.append(x)
        return torch.stack(res)

    def ProgramEncoder(self, program, S = None, tensor = None):
        '''
        Outputs a tensor of dimension the number of transitions in the CFG
        with 1 for the transitions used to derive the program and 
        0 for the others
        '''
        if S == None:
            S = self.cfg.start

        if tensor == None:
            tensor = torch.zeros(self.output_dimension)

        if isinstance(program, Function):
            F = program.function
            args_P = program.arguments
            tensor[self.RuleToIndex[(S, F)]] = 1
            for i, arg in enumerate(args_P):
                self.ProgramEncoder(arg, self.cfg.rules[S][F][i], tensor)

        if isinstance(program, (BasicPrimitive, Variable)):
            tensor[self.RuleToIndex[(S, program)]] = 1

        assert(tensor.size() == (self.output_dimension,))
        return tensor

    def custom_collate(self, batch):
        return [batch[i][0] for i in range(len(batch))], torch.stack([batch[i][1] for i in range(len(batch))])

    # def custom_collate_2(batch):
    #     return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]

    def forward_grammar(self, list_IOs):
        """
        WORK HERE
        """
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
        # Average along any column for any 2d matrix in the batch
            x = torch.mean(x, -2)
            x = self.final_layer(x)  # get the predictions
        # reconstruct the grammar
            grammars = []
            rules = {}
            for S in self.cfg.rules:
                rules[S] = {}
                for P in self.cfg.rules[S]:
                # x[self.IOEncoder.RuleToIndex[(S,P)] is the predicted proba of the rule S -> P
                    rules[S][P] = self.cfg.rules[S][P], float(
                        x[self.IOEncoder.RuleToIndex[(S, P)]])
            grammars.append(rules)
            res.append(PCFG(cfg.start, rules, max_program_depth=cfg.max_program_depth))

        # ?TODO? maybe normalize the grammar before outputing it??
        # return grammars
        return res










