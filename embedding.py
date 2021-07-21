import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 cuda=False,
                 # lexicon is the list of symbols that can occur 
                 # in the inputs and outputs
                 lexicon=None,
                 # how many hidden units
                 H=32,
                 # should the recurrent units be bidirectional?
                 bidirectional=False):
        super(RecurrentFeatureExtractor, self).__init__()

        assert lexicon
        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDING",  # ending of entire sequence
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDOFINPUT"  # delimits the ending of an input - we might have multiple inputs
        ]
        lexicon += self.specialSymbols
        encoder = nn.Embedding(len(lexicon), H)
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(H, H, layers, bidirectional=bidirectional)
        self.model = model

        self.use_cuda = cuda
        self.lexicon = lexicon
        self.symbolToIndex = {
            symbol: index for index,symbol in enumerate(lexicon)
            }
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]

        # Maximum number of inputs/outputs we will run the recognition
        # model on per task
        # This is an optimization hack
        self.MAXINPUTS = 100

        if cuda: self.cuda()

    @property
    def output_dimensionality(self): return self.H

    # modify examples before forward (to turn them into iterables of lexicon)
    # you should override this if needed
    def tokenize(self, x): return x

    def packExamples(self, examples):
        """
        IMPORTANT! xs must be sorted in decreasing order of size 
        because pytorch is stupid
        """
        es = []
        sizes = []
        for xs, y in examples:
            e = [self.startingIndex]
            for x in xs:
                for s in x:
                    e.append(self.symbolToIndex[s])
                e.append(self.endOfInputIndex)
            e.append(self.startOfOutputIndex)
            for s in y:
                e.append(self.symbolToIndex[s])
            e.append(self.endingIndex)
            if es != []:
                assert len(e) <= len(es[-1]), \
                    "Examples must be sorted in decreasing order of their tokenized size. This should be transparently handled in recognition.py, so if this assertion fails it isn't your fault as a user of EC but instead is a bug inside of EC."
            es.append(e)
            sizes.append(len(e))

        m = max(sizes)
        # padding
        for j, e in enumerate(es):
            es[j] += [self.endingIndex] * (m - len(e))

        x = torch.tensor(es)
        if self.use_cuda: x = x.cuda()
        x = self.encoder(x)
        # x: (batch size, maximum length, E)
        x = x.permute(1, 0, 2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    def examplesEncoding(self, examples):
        examples = sorted(examples, key=lambda xs_y: sum(
            len(z) + 1 for z in xs_y[0]) + len(xs_y[1]), reverse=True)
        x, sizes = self.packExamples(examples)
        outputs, hidden = self.model(x)

        # I don't know whether to return the final output or the final hidden
        # activations...
        return hidden[0, :, :] + hidden[1, :, :]

    def forward_one_task(self, examples):
        tokenized = self.tokenize(examples)

        if hasattr(self, 'MAXINPUTS') and len(tokenized) > self.MAXINPUTS:
            tokenized = list(tokenized)
            random.shuffle(tokenized)
            tokenized = tokenized[:self.MAXINPUTS]
        e = self.examplesEncoding(tokenized)

        # max pool
        # e,_ = e.max(dim = 0)

        # take the average activations across all of the examples
        # better because we might be testing on data
        # which has far more or far fewer examples than training
        e = e.mean(dim=0)
        return e

    def forward(self, tasks):
        """tasks: list of tasks
        each task is a list of I/O
        each I/O is a tuple of input, output
        each output is a list whose members are elements of self.lexicon
        each input is a tuple of lists, and each member of each such list is an element of self.lexicon

        returns: tensor of shape [len(tasks),self.output_dimensionality]
        """
        return torch.stack([self.forward_one_task(task) for task in tasks])
