import logging
import argparse
import matplotlib.pyplot as plt
import torch 

from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from Predictions.dataset_sampler import Dataset
from Predictions.IOencodings import FixedSizeEncoding, VariableSizeEncoding
from Predictions.embeddings import SimpleEmbedding, RNNEmbedding
from Predictions.models import GlobalRulesPredictor, LocalRulesPredictor

import dsl
from DSL.deepcoder import semantics, primitive_types

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

############################
##### Hyperparameters ######
############################

max_program_depth = 4

size_max = 5 # maximum number of elements in a list (input or output)
nb_inputs_max = 5 # maximum number of inputs in an IO
lexicon = list(range(30)) # all elements of a list must be from lexicon
# only useful for VariableSizeEncoding
encoding_output_dimension = 15 # fixing the dimension

embedding_output_dimension = 5
# only useful for RNNEmbedding
number_layers_RNN = 1

size_hidden = 64
nb_epochs = 2

dataset_size = 10_000
batch_size = 500

############################
######### PCFG #############
############################

deepcoder = dsl.DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
deepcoder_cfg = deepcoder.DSL_to_CFG(type_request, max_program_depth = max_program_depth)
deepcoder_pcfg = deepcoder_cfg.CFG_to_Uniform_PCFG()

############################
###### IO ENCODING #########
############################

# IO = [[I1, ...,Ik], O]
# I1, ..., Ik, O are lists
# IOs = [IO1, IO2, ..., IOn]
# task = (IOs, program)
# tasks = [task1, task2, ..., taskp]

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

print("IOEncoder.output_dimension", IOEncoder.output_dimension)

# IOEncoder = VariableSizeEncoding(
#     nb_inputs_max = nb_inputs_max,
#     lexicon = lexicon,
#     output_dimension = encoding_output_dimension,
#     )

############################
######### EMBEDDING ########
############################

# IOEmbedder = SimpleEmbedding(
#     IOEncoder = IOEncoder,
#     output_dimension = embedding_output_dimension,
#     size_hidden = size_hidden, 
#     )

IOEmbedder = RNNEmbedding(
    IOEncoder = IOEncoder,
    output_dimension = embedding_output_dimension,
    size_hidden = size_hidden, 
    number_layers_RNN = number_layers_RNN,
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
    size_hidden = size_hidden, 
    )

# model = LocalRulesPredictor(
#     cfg = deepcoder_cfg, 
#     IOEncoder = IOEncoder,
#     IOEmbedder = IOEmbedder,
#     size_hidden = size_hidden, 
#     )

loss = model.loss
optimizer = model.optimizer
ProgramEncoder = model.ProgramEncoder

############################
######## DATASET ###########
############################

dataset = Dataset(
    size = dataset_size,
    dsl = deepcoder, 
    pcfg = deepcoder_pcfg, 
    nb_inputs_max = nb_inputs_max,
    arguments = type_request.arguments(),
    # IOEncoder = IOEncoder,
    # IOEmbedder = IOEmbedder,
    ProgramEncoder = ProgramEncoder,
    size_max = size_max,
    lexicon = lexicon,
    )

dataloader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    collate_fn = model.custom_collate,
    )

############################
######## TRAINING ##########
############################

def train():
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

def print_embedding():
    print(IOEmbedder.embedding.weight)
    print([x for x in IOEmbedder.embedding.weight[:,0]])
    x = [x for x in IOEmbedder.embedding.weight[:,0]]
    y = [x for x in IOEmbedder.embedding.weight[:,1]]
    label = [str(a) for a in lexicon]
    plt.plot(x,y, 'o')
    for i, s in enumerate(label):
        xx = x[i]
        yy = y[i]
        plt.annotate(s, (xx, yy), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()

def test():
    (batch_IOs, batch_program) = next(dataloader)
    batch_predictions = model(batch_IOs)
    batch_grammars = model.reconstruct_grammars(batch_predictions)
    for program, grammar in zip(batch_program, batch_grammars):
        # print("predicted grammar {}".format(grammar))
        print("intended program {}\nprobability {}".format(
            program, grammar.probability_program(model.cfg.start, program)))

train()
test()
print_embedding()
