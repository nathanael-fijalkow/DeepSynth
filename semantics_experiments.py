import torch
import pickle
import glob
import os

from run_experiment import gather_data, list_algorithms
from DSL.list import semantics, primitive_types
from dsl import DSL
from type_system import BOOL, Arrow, List, INT, Type
from Predictions.IOencodings import FixedSizeEncoding, VariableSizeEncoding
from Predictions.embeddings import SimpleEmbedding, RNNEmbedding
from Predictions.models import GlobalRulesPredictor, LocalRulesPredictor


## START OF MODEL CREATION


############################
##### Hyperparameters ######
############################

max_program_depth = 4

size_max = 10  # maximum number of elements in a list (input or output)
nb_inputs_max = 2  # maximum number of inputs in an IO
lexicon = list(range(30))  # all elements of a list must be from lexicon
# only useful for VariableSizeEncoding
encoding_output_dimension = 30  # fixing the dimension

embedding_output_dimension = 10
# only useful for RNNEmbedding
number_layers_RNN = 1

size_hidden = 64

############################
######### PCFG #############
############################

deepcoder = DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
deepcoder_cfg = deepcoder.DSL_to_CFG(
    type_request, max_program_depth=max_program_depth)
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
    nb_inputs_max=nb_inputs_max,
    lexicon=lexicon,
    size_max=size_max,
)


# IOEncoder = VariableSizeEncoding(
#     nb_inputs_max = nb_inputs_max,
#     lexicon = lexicon,
#     output_dimension = encoding_output_dimension,
#     )

print("IOEncoder.output_dimension", IOEncoder.output_dimension)

############################
######### EMBEDDING ########
############################

IOEmbedder = SimpleEmbedding(
    IOEncoder=IOEncoder,
    output_dimension=embedding_output_dimension,
    size_hidden=size_hidden,
)

IOEmbedder = RNNEmbedding(
    IOEncoder=IOEncoder,
    output_dimension=embedding_output_dimension,
    size_hidden=size_hidden,
    number_layers_RNN=number_layers_RNN,
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
    cfg=deepcoder_cfg,
    IOEncoder=IOEncoder,
    IOEmbedder=IOEmbedder,
    size_hidden=size_hidden,
)

# model = LocalRulesPredictor(
#     cfg = deepcoder_cfg,
#     IOEncoder = IOEncoder,
#     IOEmbedder = IOEmbedder,
#     # size_hidden = size_hidden,
#     )

loss = model.loss
optimizer = model.optimizer

MODEL_NAME = ""
if isinstance(IOEncoder, FixedSizeEncoding):
    MODEL_NAME += "fixed"
else:
    MODEL_NAME += "variable"
if isinstance(IOEmbedder, SimpleEmbedding):
    MODEL_NAME += "+simple"
else:
    MODEL_NAME += "+rnn"
if isinstance(model, LocalRulesPredictor):
    MODEL_NAME += "+local"
else:
    MODEL_NAME += "+global"
print("Training model:", MODEL_NAME)

if os.path.exists("./" + MODEL_NAME + "_ongoing.weights"):
    model.load_state_dict(torch.load("./" + MODEL_NAME + "_ongoing.weights"))
    print("Loaded weights")

ProgramEncoder = model.ProgramEncoder

## END OF MODEL CREATION

# Load all tasks
tasks = []
for file in glob.glob("./list_dataset/*.pickle"):
    with open(file, "rb") as fd:
        (name, examples) = pickle.load(fd)
        tasks.append((name, examples))
print("Loaded", len(tasks), "tasks")


def _get_type(el, fallback=None):
    if isinstance(el, bool):
        return BOOL
    elif isinstance(el, int):
        return INT
    elif isinstance(el, list):
        if len(el) > 0:
            return List(_get_type(el[0]))
        else:
            return _get_type(fallback[0], fallback[1:])
    elif isinstance(el, tuple):
        assert el[-1] == None
        return _get_type(el[0], el[1:-1])
    assert False, f"Unknown type for:{el}"


def get_type_request(examples):
    input, output = examples[0]
    return Arrow(_get_type(input[0], [i[0] for i, o in examples[1:]]), _get_type(output, [o for i, o in examples[1:]]))


## Actual experiment

def make_checker(dsl, examples):
    def checker(program):
        for example in examples:
            input, output = example
            out = program.eval_naive(dsl, input)
            if output != out:
                return False
        return True
    return checker


dataset = []
for task in tasks:
    name, examples = task
    type_request: Type = get_type_request(examples)
    target_type = type_request.returns()
    if isinstance(model, GlobalRulesPredictor) and type_request != Arrow(List(INT), List(INT)):
        continue

    examples = [(i, o) for i, o in examples if len(i[0]) <= IOEncoder.size_max and all(
        [el in IOEncoder.symbolToIndex for el in i[0]]) and all([el in IOEncoder.symbolToIndex for el in o])]
    if len(examples) == 0:
        continue

    dsl = DSL(semantics, primitive_types)
    try:
        ex = [[([i[0]], o) for i, o in examples]]
        grammar = model(ex)[0]
    except AssertionError as e:
        continue
    if not isinstance(model, LocalRulesPredictor):
        grammar = model.reconstruct_grammars([grammar])[0]

    dataset.append((name, grammar, make_checker(dsl, examples)))


# Data gathering
for algo_index in range(len(list_algorithms)):
    if os.path.exists(f"./algo_{algo_name}_results_semantic.pickle"):
        continue
    data = gather_data(dataset, algo_index)
    algo_name = list_algorithms[algo_index][1]
    with open(f"./algo_{algo_name}_results_semantic.pickle", "wb") as fd:
        pickle.dump(data, fd)
