
import os
import typing
import torch
from type_system import INT, STRING, Arrow, List
from typing import Dict, Tuple, Type
from cfg import CFG
from dsl import DSL
from DSL import list, deepcoder, flashfill
from Predictions.IOencodings import FixedSizeEncoding
from Predictions.embeddings import RNNEmbedding, SimpleEmbedding
from Predictions.models import RulesPredictor, BigramsPredictor, NNDictRulesPredictor


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
    if isinstance(model, NNDictRulesPredictor):
        name += "+local"
    elif isinstance(model, RulesPredictor):
        name += "+global"
    else:
        name += "+local_bigram"
    return name


def __buildintlist_model(dsl: DSL, max_program_depth: int, nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> Tuple[CFG, RulesPredictor]:
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
        nb_arguments_max=nb_arguments_max,
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

    model = RulesPredictor(
        cfg=cfg,
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        size_hidden=size_hidden,
    )


    return cfg, model


def build_dreamcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, RulesPredictor]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 1  # maximum number of inputs in an IO
    lexicon = [x for x in range(-30, 30)]  # all elements of a list must be from lexicon

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64


    dreamcoder = DSL(list.semantics, list.primitive_types)

    dreamcoder_cfg, model = __buildintlist_model(
        dreamcoder, max_program_depth, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)
    
    if autoload:
        weights_file = get_model_name(model) + "_dreamcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return dreamcoder, dreamcoder_cfg, model


def build_deepcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, RulesPredictor]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 1  # maximum number of inputs in an IO
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 256)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_cfg, model = __buildintlist_model(
        deepcoder_dsl, max_program_depth, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, deepcoder_cfg, model


def __build_generic_model(dsl: DSL, cfg_dictionary: Dict[Type, CFG], nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> BigramsPredictor:
    IOEncoder = FixedSizeEncoding(
        nb_arguments_max=nb_arguments_max,
        lexicon=lexicon,
        size_max=size_max,
    )
    IOEmbedder = RNNEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    return BigramsPredictor(
        cfg_dictionary=cfg_dictionary,
        primitive_types={x: x.type for x in dsl.list_primitives},
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
    )


def build_deepcoder_generic_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, BigramsPredictor]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 1
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 256)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_dsl.instantiate_polymorphic_types()
    requests = deepcoder_dsl.all_type_requests(nb_arguments_max)
    cfg_dict = {}
    for type_req in requests:
        # Skip if it contains a list list
        if any(ground_type.size() >= 3 for ground_type in type_req.list_ground_types()):
            continue
        # Why the try?
        # Because for request type: int -> list(list(int)) in a DSL without a method to go from int -> list(int)
        # Then there is simply no way to produce the correct output type
        # Thus when we clean the PCFG by removing useless rules, we remove the start symbol thus creating an error
        try:
            cfg_dict[type_req] = deepcoder_dsl.DSL_to_CFG(
                type_req, max_program_depth=max_program_depth)
        except:
            continue
    print("Requests:", cfg_dict.keys())

    model = __build_generic_model(
        deepcoder_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, cfg_dict, model


def build_flashfill_generic_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[DSL, CFG, BigramsPredictor]:
    from flashfill_dataset_loader import get_lexicon
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 3
    # all elements of a list must be from lexicon
    lexicon = get_lexicon()

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    flashfill_dsl = DSL(flashfill.semantics,
                        flashfill.primitive_types, flashfill.no_repetitions)

    flashfill_dsl.instantiate_polymorphic_types()
    requests = flashfill_dsl.all_type_requests(nb_arguments_max)
    cfg_dict = {}
    for type_req in requests:
        # Skip if it contains a list list
        if any(ground_type.size() >= 3 for ground_type in type_req.list_ground_types()):
            continue
        if any(arg != STRING for arg in type_req.arguments()):
            continue
        # Why the try?
        # Because for request type: int -> list(list(int)) in a DSL without a method to go from int -> list(int)
        # Then there is simply no way to produce the correct output type
        # Thus when we clean the PCFG by removing useless rules, we remove the start symbol thus creating an error
        try:
            cfg_dict[type_req] = flashfill_dsl.DSL_to_CFG(
                type_req, max_program_depth=max_program_depth)
        except:
            continue
    print("Requests:", cfg_dict.keys())

    model = __build_generic_model(
        flashfill_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_flashfill.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return flashfill_dsl, cfg_dict, model
