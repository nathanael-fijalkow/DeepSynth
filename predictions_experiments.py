import logging
import argparse
import matplotlib.pyplot as plt
import torch 

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from Predictions.dataset_sampler import Dataset
from Predictions.IOencodings import FixedSizeEncoding, VariableSizeEncoding
from Predictions.embeddings import SimpleEmbedding#, RecurrentEmbedding
from Predictions.models import GlobalRulesPredictor

import dsl
from DSL.deepcoder import semantics, primitive_types

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])


############################
######### PCFG #############
############################

deepcoder = dsl.DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
deepcoder_cfg = deepcoder.DSL_to_CFG(type_request, max_program_depth = 4)
deepcoder_pcfg = deepcoder_cfg.CFG_to_Random_PCFG(alpha=0.7)

############################
###### IO ENCODING #########
############################

# IO = [[I1, ...,Ik], O]
# I1, ..., Ik, O are lists
# IOs = [IO1, IO2, ..., IOn]
# task = (IOs, program)
# tasks = [task1, task2, ..., taskp]

size_max = 5 # maximum number of elements in a list (input or output)
nb_inputs_max = 5 # maximum number of inputs in an IO
lexicon = list(range(30)) # all elements of a list must be from lexicon

#### Specification: #####
# IOEncoder.output_dimension: size of the encoding of one IO
# IOEncoder.lexicon_size: size of the lexicon
# IOEncoder.encode_IO: outputs a tensor of dimension IOEncoder.output_dimension
# IOEncoder.encode_IOs: inputs a list of IO of size n
# and outputs a tensor of dimension n * IOEncoder.output_dimension

IOEncoder = FixedSizeEncoding(
    nb_inputs_max = nb_inputs_max,
    lexicon = lexicon,
    size_max = size_max,
    )

# # only useful for VariableSizeEncoding
# encoding_output_dimension = 15 # fixing the dimension, 

# IOEncoder2 = VariableSizeEncoding(
#     nb_inputs_max = nb_inputs_max,
#     lexicon = lexicon,
#     output_dimension = encoding_output_dimension,
#     )

############################
######### EMBEDDING ########
############################

embedding_output_dimension = 5

IOEmbedder = SimpleEmbedding(
    IOEncoder = IOEncoder,
    output_dimension = embedding_output_dimension,
    )

#### Specification: #####
# IOEmbedder.output_dimension: size of the output of the embedder
# IOEmbedder.forward_IOs: inputs a list of IOs
# and outputs the embedding of the encoding of the IOs
# which is a tensor of dimension
# (IOEmbedder.input_dimension, IOEmbedder.output_dimension)
# IOEmbedder.forward: same but with a batch of IOs

############################
######### MODEL ############
############################

model = GlobalRulesPredictor(
    cfg = deepcoder_cfg, 
    IOEncoder = IOEncoder,
    IOEmbedder = IOEmbedder,
    size_hidden = 64, 
    )

loss = model.loss
optimizer = model.optimizer
nb_epochs = model.nb_epochs
ProgramEncoder = model.ProgramEncoder

############################
######## DATASET ###########
############################

dataset_size = 10_000

dataset = Dataset(
    size = dataset_size,
    dsl = deepcoder, 
    pcfg = deepcoder_pcfg, 
    nb_inputs_max = nb_inputs_max,
    arguments = type_request.arguments(),
    IOEncoder = IOEncoder,
    IOEmbedder = IOEmbedder,
    ProgramEncoder = ProgramEncoder,
    size_max = size_max,
    lexicon = lexicon,
    )

batch_size = 500

dataloader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    collate_fn = model.custom_collate,
    )

############################
######## TRAINING ##########
############################

# print("IOEncoder.output_dimension", IOEncoder.output_dimension)
# print("IOEmbedder.output_dimension", IOEmbedder.output_dimension)

for epoch in range(nb_epochs):
    for (batch_IOs, batch_program) in dataloader:
        optimizer.zero_grad()
        # print("batch_program", batch_program.size())
        batch_predictions = model(batch_IOs)
        # print("batch_predictions", batch_predictions.size())

        loss_value = loss(batch_predictions, batch_program)
        loss_value.backward()
        optimizer.step()

    print("epoch: {}\t loss: {}".format(epoch, float(loss_value)))

# print(model.embed.weight)
# print([x for x in model.embed.weight[:,0]])
# x = [x for x in model.embed.weight[:,0]]
# y = [x for x in model.embed.weight[:,1]]
# label = [str(a+min_int) for a in range(len(x))]
# plt.plot(x,y, 'o')
# for i, s in enumerate(label):
#     xx = x[i]
#     yy = y[i]
#     plt.annotate(s, (xx, yy), textcoords="offset points", xytext=(0,10), ha='center')
# plt.show()

# def test(self, programs, tasks):
#     for task, program in zip(tasks, programs):
#         grammar = self.forward_grammar(
#             self.IOEncoder.embed_all_examples(task))
#         # grammar = grammar.normalise()
#         # program = self.IOEncoder.embed_program(program)
#         # print("predicted grammar {}".format(grammar))
#         print("intended program {}\nprobability {}".format(
#             program, grammar.probability_program(self.cfg.start, program)))
