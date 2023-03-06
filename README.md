# DeepSynth

DeepSynth is a general-purpose program synthesizer in the programming by example framework: the user provides a few examples as pairs of input and output, DeepSynth finds a program matching the examples.

See the official [webpage](https://deepsynth.labri.fr/)

This is the repository for the code of the paper **"[Scaling Neural Program Synthesis with Distribution-based Search](https://arxiv.org/abs/2110.12485)"**
published in the conference proceedings of the AAAI Conference on Artificial Intelligence, AAAI'22 and selected for Oral Presentation.

The code was published in the **[Journal of Open Source Software](https://doi.org/10.21105/joss.04151)**.

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
pip install -r requirements.txt
# or to do it manually
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

- [Creating a DSL](#creating-a-dsl)
- [Compiling a DSL into a CFG](#compiling-a-dsl-into-a-cfg)
- [From a CFG to a PCFG](#from-a-cfg-to-a-pcfg)
- [Synthesis](#synthesis)
- [Creating a model](#model-creation)
- [Training a model](#model-training)
- [Predictions from a model](#predictions-from-a-model)

### Creating a DSL

The folder ``DSL`` contains some Domain-Specific-Languages (DSL), you can use these or create your own.
The DSL we use for the dreamcoder dataset is based on the DeepCoder paper, it is defined in ``DSL/deepcoder.py``.
It contains two important objects: ``primitive_types`` and ``semantics``.
The former (``primitive_types``) is a dictionary mapping the primitive names of the DSL to their types, and the latter (``semantics``) maps the primitive names to their semantics, which are either values or functions to be used when evaluating the primitives.
Note that a primitive can be simply a constant such as 0, 1, 2 (see the ``list.py`` DSL for an example of that).

Let us first discuss the type system, imported with ``from type_system import *``.
The atomic types called ``PrimitiveType`` are ``INT`` and ``BOOL``, you can create your own with ``PrimitiveType(type_name)``.
There are two constructors: ``List`` constructs lists by ``List(type)``, and ``Arrow(a, b)`` represents a function from ``a`` to ``b``. 
For instance, the type of ``+`` is ``Arrow(INT, Arrow(INT, INT))``. Note that this is different from ``Arrow(Arrow(INT, INT), INT)``.
We support polymorphic types: ``PolymorphicType(name)``. 
For example with ``t0 = PolymorphicType("t0"), t1 = PolymorphicType("t1")``, the type ``Arrow(t0, Arrow(t1), t0))`` can be instantiated into ``Arrow(INT, Arrow(INT), INT))`` but not into a ``Arrow(INT, Arrow(INT), BOOL))`` since both ``t0`` must correspond to the same type.

One important point: when defining the semantics of functions, Python must be able to evaluate them one argument at a time: for instance, for addition: ``lambda a: lambda b: a + b``, not ``lambda a,b: a + b``.

The optional ``no_repetitions`` is a set of primitives than cannot be repeated such as ``SORT`` (indeed it is useless to sort a sorted list). This reduces the program space.

To create a DSL object we use the following:
``dsl = DSL(primitive_types, semantics, no_repetitions)`` (``no_repetitions`` can be ``None``).


### Compiling a DSL into a CFG

We can build a Context-Free Grammar (CFG) from a DSL, using the following:
``dsl.DSL_to_CFG(type_request, 
    max_program_depth=4,
    min_variable_depth=1,
    upper_bound_type_size=10,
    n_gram=2)``.
The only required field is ``type_request``, it defines the type of the program you want to synthesize. 
For example if the goal is to synthesize the multiplication function that the type request would be ``Arrow(INT, Arrow(INT, INT))``.

The ``max_program_depth: int`` gives an upper bound on the depth of the programs.
The ``min_variable_depth: int`` gives a lower bound on the depth of all variables in a program.
The ``upper_bound_type_size: int`` is used when instantitating polymorphic types. The type size is defined as follows: a ``PrimitiveType`` has size 1, a ``List(a)`` has size ``1 + size(a)`` and an ``Arrow(a, b)`` has size ``1 + size(a) + size(b)``.
The ``n_gram: int`` chooses the granularity of the CFG, meaning how many primitives are used to choose the next primitive. The two expected values are ``1`` and ``2``. 

### From a CFG to a PCFG

To turn a CFG into a Probabilistic CFG (PCFG), we need to add probabilities to derivation rules.
The expected way to do that is by training a neural network, we'll discuss that later.
For testing purposes there are two easier ways to get a PCFG: 
* the uniform PCFG can be obtained using ``cfg.CFG_to_Uniform_PCFG()``
* a random PCFG with ``cfg.CFG_to_Random_PCFG(alpha=0.7)``. The parameter ``alpha`` serves as temperature: the larger the closer to ``uniform``.

To see some programs generated by the ``PCFG`` you can run the following code:

```python
for program in pcfg.sampling():
    print(program)
# Note that this is an infinite loop
```

### Synthesis

We can now solve a programming by example task. 
Let us consider the following: the type request is ``Arrow(INT, Arrow(INT, INT))`` and the set of examples

```python
examples = [
    ([1, 1], 5),
    ([10, 0], 13),
    ([0, 2], 1),
]
```
A solution program is ``lambda x: lambda y: x - y + 3``, there could be others (more complicated).

Following the above let us assume we have constructed a ``DSL`` and a ``PCFG``. 
We will use the following function from ``run_experiment.py``:

```python
run_algorithm(is_correct_program: Callable[[Program, bool], bool], 
    pcfg: PCFG, 
    algo_index: int)
```

Let us explain what are the three arguments:
* ``is_correct_program`` is a function that checks if the program is correct, it can be easily created with the help of ``experiment_helper.py`` which provides the following function:

```python
make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], bool]
```

* ``pcfg`` is the ``PCFG`` 
* ``algo_index`` is the index of the algorithm to use for the search. Here is the mapping:

```python
0 => Heap Search
1 => SQRT
2 => Threshold
3 => Sort & Add
4 => DFS
5 => BFS
6 => A*
```

We can further tune three parameters in ``run_experiment.py``:
* ``timeout: int = 100`` is the timeout in seconds before the search is stopped
* ``total_number_programs: int = 1_000_000`` maximum number of programs enumerated before the search is stopped. On a personal computer this is about 30sec of search for Heap Search.
* ``use_heap_search_cached_eval = True`` only when using ``Heap Search`` this enables caching of evaluations of partial programs and thus provides a much faster evaluation at the cost of additional memory.

Now, the ``run_experiment`` returns a tuple: ``program, search_time, evaluation_time, nb_programs, cumulative_probability, probability``.
Times are in seconds. Program is ``None`` is no solution was found.
``probability`` is the probability of the latest program enumerated if no solution was found and the probability of the solution program otherwise.

#### Parallelisation

There is in ``run_experiment.py`` the parallel variant ``run_algorithm_parallel`` which takes additional arguments:

- ``splits: int`` the number of splits of the grammar
- ``n_filters: int = 4`` the number of evaluators threads (a CPU per evaluator is required)
- ``transfer_queue_size: int = 500_000`` the size of the queue between enumerators and evaluators
- ``transfer_batch_size: int = 10`` the size of batches transferred from enumerators to the queue and from the queue to evaluators

The output is the same, except for some metrics which are now in list form which means they are per enumerator or per evaluator.

Up to now we did not use machine learned models. Let us get to the machine learning part.

### Model Creation

The file ``model_loader.py`` contains generic functions for the two types of models: int list models and the generic models.
The former support only one type request while the latter support generic type requests.

For instance, the function ``build_deepcoder_generic_model`` creates a ``BigGramsPredictor``.
There are a few parameters for our model:

- ``nb_arguments_max`` is the maximum number of arguments a function can have
- ``lexicon`` is the list of all symbols that can be encountered in examples, it helps us with the first step to create an integer encoding of the examples
- ``embedding_output_dimension`` the embedding size of an example, then for each example they are embedded using a classic torch embedding
- ``size_hidden`` is the sizes of the inner layers and output layers of the RNN which is run on all examples of a task sequentially
- ``number_layers_RNN`` is the number of layers the RNN should have

### Model training

The reference file is ``produce_network.py``.
After loading a model, it generates valid tasks with their solution for the model and train the model on these tasks.
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
        _, type_requests = \
            deepcoder_dataset_loader.load_tasks("./deepcoder_dataset/T=3_test.json")
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

### Predictions from a model

Now we have a model, let us see how to use it to construct a PCFG.
This is model dependent, but the good news is that the function ``task_set2dataset`` from ``experiment_helper.py`` does it for us.
The arguments are:

- ``tasks`` a list where for each task there is a list of tuples (input, output) which are the examples of the task
- ``model`` is the model
- ``dsl`` is the DSL

The output is for each task the tuple ``(task_name, PCFG_predicted_by_model, is_correct_program)``.

## Reproducing the experiments from the AAAI'22 paper

For the experiments, you only need to run the `produce_network.py` file (editing the parameters based on the dataset or the batch size).
A ```.weights``` file should appear at the root folder.
This will train a neural network on random generated programs as described in Appendix F in the paper.

All of the files mentioned in this section are located in the root folder and follow this pattern ```run_*_experiments*.py```.
Here is a short summary of each experiment:

- ```run_random_PCFGsearch.py``` produce a list of all programs generated under Xsec of search time by all algorithms.
- ```run_random_PCFGsearch_parallel.py``` same experiment but iwth the grammar_splitter and multiple CPUs.
- ```run_experiments_<dataset>.py``` try to find solutions using an ANN to predict the grammar and for each algorithm logs the search data for the corresponding ```<dataset>```. The suffix ```parallel``` can also be found indicating that the algorithms are run in parallel. The semantics experiments in the paper used a trained model thatn can be obtained using ```produce_network.py``` or directly in the repository. The results can be plotted using ```plot_results_semantics.py```.

Note that for the DreamCoder experiment in our paper, we did not use the cached evaluation of HeapSearch, this can be reproduced by setting ```use_heap_search_cached_eval``` to ```False``` in ```run_experiment.py```.

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
