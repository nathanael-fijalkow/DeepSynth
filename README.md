# DeepSynth
General-purpose program synthesiser.

This is the repository for the code of the paper **"Scaling Neural Program Synthesis with Distribution-based Search"** submitted at AAAI 2021.
[Link to paper](TODO)

**Authors**:
Anonymous


![Figure](https://github.com/nathanael-fijalkow/DeepSynth/raw/main/main_figure.png)

### Abstract

 _We consider the problem of automatically constructing computer programs from input-output examples. We investigate
how to augment probabilistic and neural program synthesis methods with new search algorithms, proposing a framework called distribution-based search. Within this framework,
we introduce two new search algorithms: HEAP SEARCH,
an enumerative method, and SQRT SAMPLING, a probabilistic method. We prove certain optimality guarantees for
both methods, show how they integrate with probabilistic
and neural techniques, and demonstrate how they can operate
at scale across parallel compute environments. Collectively
these findings offer theoretical and applied studies of search
algorithms for program synthesis that integrate with recent
developments in machine-learned program synthesizers._

## Usage

### Installation

```bash
# clone this repository

# create your new env
conda create -n deep_synth
# activate it
conda activate deep_synth
# install pip
yes | conda install pip
# install this package and the dependencies
pip install torch cython tqdm numpy ray
pip install git+https://github.com/Theomat/sbsur
pip install git+https://github.com/MaxHalford/vose


# You are good to go :)
# To test your installation you can run the following tests:
python unit_test_algorithms.py
python unit_test_programs.py
python unit_test_algorithms.py
python unit_test_predictions.py
python unit_test_parallel.py
```

### File structure

```bash
./
        Algorithms/      # the search algorithms + parallel pipeline
        DSL/             # DSL
        list_dataset/    # DeepCoder dataset in pickle format
        Predictions/     # all files related to the ANN for prediction of the grammars 
```

### Reproducing the experiments

All of the files mentioned in this section are located in the root folder and ends with ```experiments```.

Here is a short summary of each experiment:

- ```syntactic_experiments.py``` produce a list of all programs geneerated under 2sec of search time by all algorithms.
- ```semantic_experiments.py``` try to find solutions using an ANN to predict the grammar and for each algorithm logs the search data for the ```list_datatset```.
- ```semantic_experiments_parallel.py``` same experiment but algorithms are run in parallel.

### Quick guide to using ANN to predict a grammar

Is it heavily inspired by the file ```semantics_experiments.py```.

First we create a prediction model:

```python
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
```

Now we can produce the grammars:

```python
dsl = DSL(semantics, primitive_types)
batched_grammars = model(batched_examples)
if isinstance(model, GlobalRulesPredictor):
    batched_grammars = model.reconstruct_grammars(batched_grammars)
```

### Quick guide to using a search algorithm for a grammar

There are alredy functions for that in ```run_experiment.py```, namely ```run_algorithm``` and ```run_algorithm_parallel```.

The fromer enables you to run the specified algorithm in a single thread while the latter in parallel with a grammar splitter.
