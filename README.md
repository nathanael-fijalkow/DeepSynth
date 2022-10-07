# DeepSynth

DeepSynth is a general-purpose program synthesizer in the programming by example framework: the user provides a few examples as pairs of input and output, DeepSynth finds a program matching the examples.

See the official [webpage](https://deepsynth.labri.fr/)

This is the repository for the code of the paper **"[Scaling Neural Program Synthesis with Distribution-based Search](https://arxiv.org/abs/2110.12485)"**
published in the conference proceedings of the AAAI Conference on Artificial Intelligence, AAAI'22 and selected for Oral Presentation.

**Authors**:
Nathanaël Fijalkow, Guillaume Lagarde, Théo Matricon, Kevin Ellis, Pierre Ohlmann, Akarsh Potta

<!-- toc -->

- [Overview](#overview)
- [Usage](#usage)
  - [Installation](#installation)
  - [File structure](#file-structure)
- [Documentation and examples](#documentation-and-examples)
    - [DSL](#dsl)
    - [CFG and PCFG](#cfg-and-pcfg)
    - [Synthesis](#synthesis)
    - [Prediction from a model](#prediction-from-a-model)
    - [Model creation](#model-creation)
    - [Model training](#model-training)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Report an issue](#report-an-issue)
- [Contribute](#contribute)

<!-- tocstop -->

## Overview

DeepSynth is a tool for automatically synthesizing programs from examples. It combines machine learning predictions with efficient enumeration techniques in a very generic way.

The following figure shows the pipeline.
![Figure](https://github.com/nathanael-fijalkow/DeepSynth/raw/main/main_figure.png)

- The **first step** is to define a domain-specific language (DSL), which is the programming language specifically designed to solve particular tasks.
- The **second** step is a compilation step: a context-free grammar (CFG) describing the set of all programs is compiled from the DSL and a number of syntactic constraints. The grammar is used to enumerate programs in an efficient way.
However the number of programs grows extremely fast with size, making program synthesis very hard computationally.
***We believe that the path to scalability is to leverage machine learning predictions in combination with efficient enumeration techniques.***
- The **third** step is to obtain predictions from the examples: a prediction model outputs predictions in the form of probabilities for the rules of the grammar, yielding a probabilistic context-free grammar (PCFG).
- The **fourth** and final step is the search: enumerating programs from the PCFG. We introduced the distribution-based search as a theoretical framework to analyse algorithms, and constructed two new algorithms: HeapSearch and SQRT Sampling.

## Usage

### Installation

```bash
# clone this repository
git clone https://github.com/nathanael-fijalkow/DeepSynth.git

# create your new env
conda create -n deep_synth "python>=3.7" 
# activate it
conda activate deep_synth
# install pip
yes | conda install pip
# install this package and the dependencies
conda install -c conda-forge cython tqdm numpy matplotlib scipy
conda install -c pytorch "pytorch>=1.8" 
pip install git+https://github.com/MaxHalford/vose
# For flashfill dataset
pip install sexpdata
# If you want to do the parallel experiments
pip install ray

# If you run in an ValueError: numpy.ufunc size changed
pip install --upgrade numpy

# You are good to go :)
# To test your installation you can run the following tests:
python unit_tests_algorithms.py
python unit_tests_programs.py
python unit_tests_predictions.py
# Only if you installed ray
python unit_tests_parallel.py
```

### File structure

```bash
./
        Algorithms/      # the search algorithms + parallel pipeline
        DSL/             # DSL: dreamcoder, deepcoder, flashfill
        list_dataset/    # DreamCoder dataset in pickle format
        Predictions/     # all files related to the ANN for prediction of the grammars 
```

## Documentation and examples

Table of contents:

- [DSL](#dsl)
- [CFG and PCFG](#cfg-and-pcfg)
- [Synthesis](#synthesis)
- [Prediction from a model](#prediction-from-a-model)
- [Model creation](#model-creation)
- [Model training](#model-training)

### DSL

For the dreamcoder dataset, we have defined the DSL in ``DSL/deepcoder.py``.
It contains two important objects: ``primitive_types`` and ``semantics``.
The former is a dictionary mapping the primitives of the DSL to their types while the latter maps the primitives to their semantics, that is a value or function to be used when evaluating the primitive.
Primitives can be constant such as 0, 1, 2... in the ``list.py`` DSL.
If they are functions, then Python must be able to apply them one argument at a time, that would give the following semantic for ``+``: ``lambda a: lambda b: a + b``.

There are two more subtleties in this file.
First, the ``no_repetitions`` is a set of primitives than cannot be repeated such as ``SORT`` because it is useless to sort a sorted list, this reduces a bit the size of the program space.

And the second are types. 
Notice that we used ``Arrow``, ``List``, ``INT``... they come from ``from type_system import *``.
There are ``PrimitiveType`` such as ``INT``, ``BOOL``, which are in truth nothing but a string, you can create your own with ``PrimitiveType(type_name)``.
They have no constraints however DeepSynth makes the difference between them.
Then there is ``List`` which makes up with ``List(type)`` a list of the given type.
``Arrow(a, b)`` represents a function from ``a`` to ``b``, the type of ``+`` is ``Arrow(INT, Arrow(INT, INT))`` which is different from ``Arrow(Arrow(INT, INT), INT)``: the former maps an int to a function that takes an int and returns an int that is a partial application of ``+`` whereas the latter takes a function mapping an int to an int and return an int.
And finally there are the ``PolymorphicType(name)``, their usage in a primitive is independent of their use in other primitives.
If used multiples times in the same type definition then they must be the same types, for example with ``t0 = PolymorphicType("t0")``, ``Arrow(t0, Arrow(PolymorphicType("t1"), t0))``, it can be instantiated into a ``INT -> INT -> INT``, ``INT -> BOOL -> INT`` but not into a ``INT -> INT -> BOOL`` both ``t0`` must correspond to the same type.

### CFG and PCFG

Now that we have a DSL, we can build a CFG.
First create a DSL object with what you defined in the previous section
``dsl = DSL(primitive_types, semantics, no_repetitions)`` (``no_repetitions`` can be ``None``).
To get a CFG, we need more information.
First, the ``type_request: Type`` which defines the type of the program you want to generate, for example I would give ``Arrow(INT, Arrow(INT, INT))`` if I wanted to generate the integer multiplication ``x`` function.
This is the only argument required to call ``dsl.DSL_to_CFG(type_request)`` and get a CFG.

However, take note of the ``max_program_depth: int`` argument which limits the maximum depth of the programs (seen as trees) produced.
If you use polymorphic types, then take note of the ``upper_bound_type_size: int`` which limits as to how big the polymorphic types can be instanced into.
The type size is simply the number of objects used in our framework to instantiate the type: a ``PrimitiveType`` is of size 1, a ``List(a)`` has size ``1 + size(a)`` and an ``Arrow(a, b)`` has size ``1 + size(a) + size(b)``.

Now we are almost to a PCFG, we only need to add probabilities to our CFG.
Just to try there are two easy ways to get a PCFG: the uniform PCFG can be obtained using ``cfg.CFG_to_Uniform_PCFG()`` and a random PCFG with ``cfg.CFG_to_Random_PCFG(alpha=0.7)`` with the bigger the ``alpha`` the closer to ``uniform`` and the lower the closer to completely random.
Normally, you would get the probabilities with a model but we'll move onto that later.

To see some programs generated by the ``PCFG`` you can run the following code:

```python
for program in pcfg.sampling():
    print(program)
# Note that this is an infinite loop
```

### Synthesis

At this point, we know how to get a ``PCFG`` for our specific ``type_request``, we would like to see if we can already synthesis a correct program.
We can import the following function from ``run_experiment.py``:

```python
run_algorithm(is_correct_program: Callable[[Program, bool], bool], pcfg: PCFG, algo_index: int) -> Tuple[Program, float, float, int, float, float]
```

The first argument is a function that checks if the program is correct, it can be easily created with the help of ``experiment_helper.py`` which provides the following function:

```python
make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], bool]
```

to which you give your DSL and the examples as a list of tuples (input, output) on which your program should be correct.
For example, let's say I want to synthesise the following function: ``lambda x: lambda y: x - y + 3``, I could give the following examples:

```python
examples = [
    ([1, 1], 5),
    ([10, 0], 13),
    ([0, 2], 1),
]
```

The second argument is our ``PCFG`` and the third argument is simply the index of the algorithm to use for the synthesis.
Here is the mapping:

```python
0 => Heap Search
1 => SQRT
2 => Threshold
3 => Sort & Add
4 => DFS
5 => BFS
6 => A*
```

They correspond to the indices of algorithms in ``list_algorithms`` in ``run_experiment.py``.
There are three additional parameters that you may want to change in ``run_experiment.py``:

- ``timeout: int = 100`` is the timeout in seconds before the search is stopped
- ``total_number_programs: int = 1_000_000`` maximum number of programs enumerated before the search is stopped, for example on my personal computer that amounts to around 30sec of search for Heap Search.
- ``use_heap_search_cached_eval = True`` only when using ``Heap Search`` this enables caching of evaluations of partial programs and thus provides a much faster evaluation at the cost of additional memory.

Now, the ``run_experiment`` returns program, search_time, evaluation_time, nb_programs, cumulative_probability, probability.
Times are in seconds. Program is None is no solution was found.
Probability is the probability of the latest program enumerated if no solution was found and the probability of the solution program otherwise.

#### Parallelisation

There is in ``run_experiment.py`` the parallel variant ``run_algorithm_parallel`` which takes additional arguments:

- ``splits: int`` the number of splits of the grammar
- ``n_filters: int = 4`` the number of evaluators threads (a CPU per evaluator is required)
- ``transfer_queue_size: int = 500_000`` the size of the queue between enumerators and evaluators
- ``transfer_batch_size: int = 10`` the size of batches transferred from enumerators to the queue and from the queue to evaluators

The output is the same, except for some metrics which are now in list form which means they are per enumerator or per evaluator.

### Prediction from a model

Let's say you have a model, how to get it to use it to get your PCFGs.
This is model dependent thankfully a function does it for us.
In ``experiment_helper.py`` there is the following function:

```python
task_set2dataset(tasks, model, dsl: DSL) -> List[Tuple[str, PCFG, Callable[[Program, bool], bool]]]
```

The arguments are:

- ``tasks`` a list where for each task there is a list of tuples (input, output) which are the examples of the task
- ``model`` is your model
- ``dsl`` is your DSL

And what you get as an output is for each task you get ``(task_name, PCFG_predicted_by_model, is_correct_program)``.
Yes it also directly computes the ``is_correct_program`` function for you.

### Model Creation

Please refer to ``model_loader.py`` which contains generic functions for the two types of models the int list models and the generic models.
The former support only one type request while the latter support generic type requests.
Here is an extract of the file:

```python
def build_deepcoder_generic_model(types: Set[Type], max_program_depth: int = 4, autoload: bool = True) -> Tuple[dsl.DSL, CFG, BigramsPredictor]:
    size_max = 19  # maximum number of elements in a list (input or output)
    nb_arguments_max = 3
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 257)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = dsl.DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_dsl.instantiate_polymorphic_types()
    cfg_dict = {}
    for type_req in types:
        cfg_dict[type_req] = deepcoder_dsl.DSL_to_CFG(type_req,
                 max_program_depth=max_program_depth)
    print("Requests:", "\n\t" + "\n\t".join(map(str, cfg_dict.keys())))

    model = __build_generic_model(
        deepcoder_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, cfg_dict, model
```

For the most part, it should be pretty self explanatory.
There are a few parameters for our model:

- ``nb_arguments_max`` is the maximum number of arguments a function can have
- ``lexicon`` is the list of all symbols that can be encountered in examples, it helps us with the first step to create an integer encoding of the examples
- ``embedding_output_dimension`` the embedding size of an example, then for each example they are embedded using a classic torch embedding
- ``size_hidden`` is the sizes of the inner layers and output layers of the RNN which is run on all examples of a task sequentially
- ``number_layers_RNN`` is the number of layers the RNN should have

### Model training

The reference here will be the ``produce_network.py`` file.
Concretely, it loads a model, then generate valid tasks with their solution for the model and train the models on these tasks.
The model is saved at each epoch.
The part you might want to change according to your needs is:

```python
dataset_name = "dreamcoder"
# dataset_name = "deepcoder"
# dataset_name = "flashfill"

# Set to None for generic model
type_request = Arrow(List(INT), List(INT))
# type_request = None

dataset_size: int = 10_000
nb_epochs: int = 1
batch_size: int = 128

## TRAINING

if dataset_name == "dreamcoder":
    cur_dsl, cfg, model = build_dreamcoder_intlist_model()
elif dataset_name == "deepcoder":
    if type_request is None:
        _, type_requests = deepcoder_dataset_loader.load_tasks("./deepcoder_dataset/T=3_test.json")
        cur_dsl, cfg_dict, model = build_deepcoder_generic_model(type_requests)
    else:
        cur_dsl, cfg, model = build_deepcoder_intlist_model()
elif dataset_name == "flashfill":
    cur_dsl, cfg_dict, model = build_flashfill_generic_model()
else:
    assert False, f"Unrecognized dataset: {dataset_name}"


print("Training model:", get_model_name(model), "on", dataset_name)
print("Type Request:", type_request or "generic")

if type_request:
    nb_examples_max: int = 2
else:
    nb_examples_max: int = 5
```

## Reproducing the experiments

All of the files mentioned in this section are located in the root folder and follow this pattern ```run_*_experiments*.py```.

Here is a short summary of each experiment:

- ```run_random_PCFGsearch.py``` produce a list of all programs generated under Xsec of search time by all algorithms.
- ```run_random_PCFGsearch_parallel.py``` same experiment but iwth the grammar_splitter and multiple CPUs.
- ```run_experiments_<dataset>.py``` try to find solutions using an ANN to predict the grammar and for each algorithm logs the search data for the corresponding ```<dataset>```. The suffix ```parallel``` can also be found indicating that the algorithms are run in parallel. The semantics experiments in the paper used a trained model thatn can be obtained using ```produce_network.py``` or directly in the repository. The results can be plotted using ```plot_results_semantics.py```.

Note that for the DreamCoder experiment in our paper, we did not use the cached evaluation of HeapSearch, this can be reproduced by setting ```use_heap_search_cached_eval``` to ```False``` in ```run_experiment.py```.

### Quick guide to train a neural network

For the experiments, you only need to run the `produce_network.py` file, do not hesitate to change the parameters to suit your needs such as the dataset or the batch size.
A ```.weights``` file should appear at the root folder.
This will train a neural network on random generated programs as described in Appendix F in the paper.

## How to download the DeepCoder dataset

First, download the archive from here (DeepCoder repo): <https://storage.googleapis.com/deepcoder/dataset.tar.gz> in a folder ```deepcoder_dataset``` at the root of **DeepSynth**.
Then you simply need to:

```bash
gunzip dataset.tar.gz
tar -xf dataset.tar
```

You should see a few JSON files.

## Report an issue

If you find any issue, please create a GitHub issue with specifics steps to reproduce the bug.

## Contribute

Contributions are welcome! However, feature-wise we believe DeepSynth is in a maintenance state.
Please first, create an issue with what your contribution should be about.
Then you can create a pull request.
