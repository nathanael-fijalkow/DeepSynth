import torch 
import logging
import argparse
import matplotlib.pyplot as plt

from type_system import Arrow, List, INT, BOOL
from Predictions.IOencodings import FixedSizeEncoding
from Predictions.models import GlobalRulesPredictor, LocalBigramsPredictor
from Predictions.dataset_sampler import Dataset

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

from model_loader import build_deepcoder_generic_model, build_deepcoder_intlist_model, build_dreamcoder_intlist_model, build_flashfill_generic_model, get_model_name


## HYPERPARMETERS

# dataset_name = "dreamcoder"
# dataset_name = "deepcoder"
dataset_name = "flashfill"

# Set to None for model invariant of type request
# type_request = Arrow(List(INT), List(INT))
type_request = None

dataset_size: int = 10_000
nb_epochs: int = 1
batch_size: int = 128

## TRAINING

if dataset_name == "dreamcoder":
    cur_dsl, cfg, model = build_dreamcoder_intlist_model()
elif dataset_name == "deepcoder":
    if type_request is None:
        cur_dsl, cfg_dict, model = build_deepcoder_generic_model()
    else:
        cur_dsl, cfg, model = build_deepcoder_intlist_model()
elif dataset_name == "flashfill":
    cur_dsl, cfg_dict, model = build_flashfill_generic_model()
else:
    assert False, f"Unrecognized dataset: {dataset_name}"


if type_request:
    nb_examples_max: int = 2
else:
    nb_examples_max: int = 5

############################
######## TRAINING ##########
############################

def train(model, dataset):
    savename = get_model_name(model) + "_" + dataset_name + ".weights"
    for epoch in range(nb_epochs):
        gen = dataset.__iter__()
        for i in range(dataset_size // batch_size):
            batch_IOs, batch_program, batch_requests = [], [], []
            for _ in range(batch_size):
                io, prog, _ , req= next(gen)
                batch_IOs.append(io)
                batch_program.append(prog)
                batch_requests.append(req)
            model.optimizer.zero_grad()
            # print("batch_program", batch_program.size())
            batch_predictions = model(batch_IOs)
            # print("batch_predictions", batch_predictions.size())
            if isinstance(model, GlobalRulesPredictor):
                loss_value = model.loss(
                    batch_predictions, torch.stack(batch_program))
            elif isinstance(model, LocalBigramsPredictor):
                batch_grammars = model.reconstruct_grammars(
                    batch_predictions, batch_requests)
                loss_value = model.loss(
                    batch_grammars, batch_program)
            loss_value.backward()
            model.optimizer.step()
            print("minibatch: {}\t loss: {}".format(i, float(loss_value)))

        print("epoch: {}\t loss: {}".format(epoch, float(loss_value)))
        torch.save(model.state_dict(), savename)

def print_embedding(model):
    print(model.IOEmbedder.embedding.weight)
    print([x for x in model.IOEmbedder.embedding.weight[:, 0]])
    x = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 0]]
    y = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 1]]
    label = [str(a) for a in model.IOEncoder.lexicon]
    plt.plot(x,y, 'o')
    for i, s in enumerate(label):
        xx = x[i]
        yy = y[i]
        plt.annotate(s, (xx, yy), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()

# def test():
#     (batch_IOs, batch_program) = next(dataloader)
#     batch_predictions = model(batch_IOs)
#     batch_grammars = model.reconstruct_grammars(batch_predictions)
#     for program, grammar in zip(batch_program, batch_grammars):
#         # print("predicted grammar {}".format(grammar))
#         print("intended program {}\nprobability {}".format(
#             program, grammar.probability_program(model.cfg.start, program)))
dataset = Dataset(
    size=dataset_size,
    dsl=cur_dsl,
    pcfg_dict={type_request: cfg.CFG_to_Uniform_PCFG()} if type_request else {
        t: cfg.CFG_to_Uniform_PCFG() for t, cfg in cfg_dict.items()},
    nb_examples_max=nb_examples_max,
    arguments={type_request: type_request.arguments()} if type_request else {
        t: t.arguments() for t in cfg_dict.keys()},
    # IOEncoder = IOEncoder,
    # IOEmbedder = IOEmbedder,
    ProgramEncoder=model.ProgramEncoder,
    size_max=model.IOEncoder.size_max,
    lexicon=model.IOEncoder.lexicon[:-2] if isinstance(
        model.IOEncoder, FixedSizeEncoding) else model.IOEncoder.lexicon[:-4],
    for_flashfill=dataset_name == "flashfill"
)


train(model, dataset)
# test()
print_embedding(model)
