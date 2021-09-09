
import os
import typing
import torch
from type_system import INT, Arrow, List
from typing import Any, Tuple
from cfg import CFG
from dsl import DSL
from DSL import list, deepcoder
from Predictions.IOencodings import FixedSizeEncoding
from Predictions.embeddings import RNNEmbedding, SimpleEmbedding
from Predictions.models import GlobalRulesPredictor, LocalRulesPredictor


def get_model_name(model) -> str:
    name: str = ""
    if isinstance(model.IOEncoder, FixedSizeEncoding):
        name += "fixed"
    else:
        name += "variable"
    if isinstance(model.IOEmbedder, SimpleEmbedding):
        name += "+simple"
    else:
        name += "+rnn"
    if isinstance(model, LocalRulesPredictor):
        name += "+local"
    else:
        name += "+global"
    return name


def __buildintlist_model(dsl: DSL, max_program_depth: int, nb_inputs_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int):
    type_request = Arrow(List(INT), List(INT))
    cfg = dsl.DSL_to_CFG(
        type_request, max_program_depth=max_program_depth)

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

    ############################
    ######### EMBEDDING ########
    ############################

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
        cfg=cfg,
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        size_hidden=size_hidden,
    )


    return cfg, model

def build_dreamcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, Any]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_inputs_max = 2  # maximum number of inputs in an IO
    lexicon = [x for x in range(-30, 30)]  # all elements of a list must be from lexicon

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64


    dreamcoder = DSL(list.semantics, list.primitive_types)

    dreamcoder_cfg, model = __buildintlist_model(
        dreamcoder, max_program_depth, nb_inputs_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)
    
    if autoload:
        weights_file = get_model_name(model) + "_dreamcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return dreamcoder, dreamcoder_cfg, model


def build_deepcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, Any]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_inputs_max = 2  # maximum number of inputs in an IO
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 256)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = DSL(deepcoder.semantics, deepcoder.primitive_types)

    deepcoder_cfg, model = __buildintlist_model(
        deepcoder_dsl, max_program_depth, nb_inputs_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, deepcoder_cfg, model
