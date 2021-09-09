import torch 
import logging
import argparse
import matplotlib.pyplot as plt

from type_system import Arrow, List, INT, BOOL
from Predictions.dataset_sampler import Dataset

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

from model_loader import build_deepcoder_intlist_model, build_dreamcoder_intlist_model, get_model_name


## HYPERPARMETERS

# dataset_name = "dreamcoder"
dataset_name = "deepcoder"

type_request = Arrow(List(INT), List(INT))

nb_epochs: int = 1
dataset_size: int = 10_000
batch_size: int  = 128

## TRAINING

if dataset_name == "dreamcoder":
    cur_dsl, cfg, model = build_dreamcoder_intlist_model()
elif dataset_name == "deepcoder":
    cur_dsl, cfg, model = build_deepcoder_intlist_model()
else:
    assert False, f"Unrecognized dataset: {dataset_name}"
############################
######## TRAINING ##########
############################

def train(model, dataset):
    savename = get_model_name(model) + "_" + dataset_name + ".weights"
    for epoch in range(nb_epochs):
        gen = dataset.__iter__()
        for i in range(dataset_size // batch_size):
            batch_IOs, batch_program = [], []
            for _ in range(batch_size):
                io, prog, _ = next(gen)
                batch_IOs.append(io)
                batch_program.append(prog)
            model.optimizer.zero_grad()
            # print("batch_program", batch_program.size())
            batch_predictions = model(batch_IOs)
            # print("batch_predictions", batch_predictions.size())
            loss_value = model.loss(
                batch_predictions, torch.stack(batch_program))
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
    pcfg=cfg.CFG_to_Uniform_PCFG(),
    nb_inputs_max=model.IOEncoder.nb_inputs_max,
    arguments=type_request.arguments(),
    # IOEncoder = IOEncoder,
    # IOEmbedder = IOEmbedder,
    ProgramEncoder=model.ProgramEncoder,
    size_max=model.IOEncoder.size_max,
    lexicon=model.IOEncoder.lexicon,
)


train(model, dataset)
# test()
print_embedding(model)
