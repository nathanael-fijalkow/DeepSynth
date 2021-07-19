import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from dsl import *
from pcfg import PCFG
from pcfg_logprob import LogProbPCFG

class RecognitionModel(nn.Module):
    def __init__(self, feature_extractor,
                 template_pcfg=None, template_dsl=None, type_request=None):
        """
        template_cfg: a cfg giving the structure that will be output 
        by this recognition model

        feature_extractor: a neural network module 
        taking a list of tasks (such as a list of input-outputs) 
        and returning a tensor of shape 
        [len(list_of_tasks), feature_extractor.output_dimensionality]

        intermediate_q: whether the pcfg weights are produced 
        by first calculating an intermediate "Q" tensor as in dreamcoder 
        and then constructing the PCFG from that
        """
        super(RecognitionModel, self).__init__()

        self.feature_extractor = feature_extractor
        
        self.intermediate_q = template_dsl is not None

        assert int(template_pcfg is None) + int(template_dsl is None) == 1,\
            "specify exactly one template: either PCFG (no q) or DSL (q)"
        
        self.template_pcfg = template_pcfg
        self.template_dsl = template_dsl

        H = self.feature_extractor.output_dimensionality # hidden
        if not self.intermediate_q:
            projection_layer = {}
            for name,productions in template_pcfg.rules.items():
                n_productions = len(productions)
                module = nn.Sequential(nn.Linear(H, n_productions),
                                       nn.LogSoftmax(-1))
                projection_layer[str(name)] = module
            self.projection_layer = nn.ModuleDict(projection_layer)
        else:
            assert type_request is not None
            self.CFG = template_dsl.DSL_to_CFG(type_request=type_request, 
                                          upper_bound_type_size=4, 
                                          max_program_depth=4, 
                                          min_variable_depth=4,
                                          n_gram = 1)

            self.number_of_primitives = len(template_dsl.list_primitives)
            self.number_of_parents = self.number_of_primitives + 1 # parent can be nothing
            self.maximum_arguments = max(len(p.type.arguments())
                                         for p in template_dsl.list_primitives )
            self.q_predictor = nn.Linear(H,
                 self.number_of_primitives*self.number_of_parents*self.maximum_arguments)
#            q = self.q_vector_to_dictionary(self.q_predictor(torch.ones(H)))
#            g = self.q2PCFG(q)

    def q_vector_to_dictionary(self,q):
        """q: size self.number_of_primitives*self.number_of_parents*self.maximum_arguments"""
        q = q.view(self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        q = nn.LogSoftmax(-1)(q)

        q_dictionary = {}
        for parent_index, parent in enumerate([None] + self.template_dsl.list_primitives):
            for a in range(self.maximum_arguments):
                for child_index, child in enumerate(self.template_dsl.list_primitives):
                    q_dictionary[parent,a,child] = q[parent_index,a,child_index]

        return q_dictionary        
            
    def q2PCFG(self,Q):
        CFG = self.CFG
        #logging.debug('CFG: %s'%format(CFG))

        augmented_rules = {}
        newQ = {}
        for S in CFG.rules:
            augmented_rules[S] = {}
            #S = (tp,(parent,my_index),?)
            _, previous, _ = S
            if previous:
                primitive, argument_number = previous
            else:
                primitive, argument_number = None, 0

            for P in CFG.rules[S]:
                found = False
                for p, a, P2 in Q:
                    if p != None and p.typeless_eq(primitive) \
                    and a == argument_number and P.typeless_eq(P2):
                        found = True
                        newQ[primitive, argument_number, P] = Q[p, a, P2]
                    if p == None and primitive == None \
                    and a == argument_number and P.typeless_eq(P2):
                        found = True
                        newQ[primitive, argument_number, P] = Q[p, a, P2]
                if not found:
                    assert False, f"not found {primitive} {argument_number} {P}"
                    newQ[primitive, argument_number, P] = 0

                augmented_rules[S][P] = \
                CFG.rules[S][P], newQ[primitive, argument_number, P]

        #logging.debug('Rules of the CFG from the initial non-terminal:\n%s'%format(augmented_rules[CFG.start]))

        return PCFG(start = CFG.start, 
                    rules = augmented_rules,
                    process_probabilities=False) # because these are tensors and logs

    def forward(self, tasks, log_probabilities=False):
        """tasks: list of tasks
        log_probabilities: if this is true then the returned PCFGs 
        will have log probabilities instead of actual probabilities, 
        and you will be able to back propagate through these long 
        probabilities. 
        otherwise it will return normal PCFGs that you can call sample on etc.

        returns: list of PCFGs
        """
        features = self.feature_extractor(tasks)
        template = self.template

        if self.intermediate_q:
            q = self.q_predictor(features)
            qs = [self.q_vector_to_dictionary(q[b])
                  for b in range(len(tasks)) ]
            gs = [self.q2PCFG(qs[b])
                  for b in range(len(tasks)) ]
            if not log_probabilities:
                # iterate over the rules and exponentiation and untorch
                rules = [ {k: { kk: (vv[0],vv[1].exp().cpu().detach().numpy())
                                for kk,vv in v.items() }
                           for k,v in g.rules.items() }
                          for g in gs ]
                gs = [PCFG(gs[0].start,r) for r in rules ]
            return gs
        
        probabilities = {name: self.projection_layer[str(name)](features)
                         for name in self.template_pcfg.rules}
        if not log_probabilities:
            probabilities = {key: value.exp().detach().cpu().numpy()
                             for key, value in probabilities.items()}

        grammars = []
        for b in range(len(tasks)): # iterate over batchs
            grammar = copy.deepcopy(self.template_pcfg)

            for name,productions in self.template_pcfg.rules.items():
                # productions is a dictionary mapping {P: (l,w) }
                # convert to a list ordered alphabetically
                # this is because the neural network outputs a vector, which is ordered
                # but the dictionaries unordered
                possible_functions = list(sorted(productions.keys(),key=str))
                for function_index, the_function in enumerate(possible_functions):
                    try:
                        l = self.template_pcfg.rules[name][the_function][0]
                    except:
                        import pdb; pdb.set_trace() 
                    grammar.rules[name][the_function] = (l, probabilities[name][b,function_index])
                    
                        

            if not log_probabilities:
                # make sure we get the right vose samplers etc.
                grammar = PCFG(grammar.start, grammar.rules,
                               max_program_depth=grammar.max_program_depth,
                               process_probabilities=False)

            grammars.append(grammar)

        probabilities = {S: self.projection_layer[format(S)](features) for S in template.rules}

        grammars = []
        for b in range(len(tasks)): # iterate over batches
            rules = {}
            for S in template.rules:
                rules[S] = {}
                for i, P in enumerate(template.rules[S]):
                    rules[S][P] = template.rules[S][P], probabilities[S][b, i]
            grammar = LogProbPCFG(template.start, 
                rules, 
                max_program_depth=template.max_program_depth)
            grammars.append(grammar)
        return grammars

class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 cuda=False,
                 # what are the symbols that can occur 
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
            symbol: index for index,
            symbol in enumerate(lexicon)
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
        # outputs, sizes = pad_packed_sequence(outputs)
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
        # I think this might be better because we might be testing on data
        # which has far more or far fewer examples then training
        e = e.mean(dim=0)
        return e

    def forward(self, tasks):
        """tasks: list of tasks
        each task is a list of input outputs
        each input output is a tuple of input, output
        each output is a list whose members are elements of self.lexicon
        each input is a tuple of lists, and each member of each such list is an element of self.lexicon

        returns: tensor of shape [len(tasks),self.output_dimensionality]"""
        #fix me! properly batch the recurrent network across all tasks at once
        return torch.stack([self.forward_one_task(task) for task in tasks])
    
if __name__ == "__main__":
    H = 128 # hidden size of neural network
    
    fe = RecurrentFeatureExtractor(lexicon=list(range(10)),
                                   H=H,
                                   bidirectional=True)

    from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
    from program import Program, Function, Variable, BasicPrimitive, New
    from dsl import DSL

    primitive_types = {
        "if": Arrow(BOOL, Arrow(INT, INT)),
        "+": Arrow(INT, Arrow(INT, INT)),
        "0": INT,
        "1": INT,
        "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
        "lt": Arrow(INT, Arrow(INT, BOOL)),
    }

    semantics = {
        "if": lambda b: lambda x: lambda y: x if b else y,
        "+": lambda x: lambda y: x + y,
        "0": 0,
        "1": 1,
        "and": lambda b1: lambda b2: b1 and b2,
        "lt": lambda x: lambda y: x <= y,
    }

    dsl = DSL(semantics, primitive_types)
    type_request = Arrow(INT, INT)
    template = dsl.DSL_to_CFG(type_request)
    model = RecognitionModel(template,fe)

    programs = [
        Function(BasicPrimitive("+", Arrow(INT, Arrow(INT, INT))),[BasicPrimitive("0", INT),BasicPrimitive("1", INT)], INT),
        BasicPrimitive("0", INT),
        BasicPrimitive("1", INT)
        ]

    xs = ([1,9],[8,8]) # inputs
    y = [3,4] # output
    ex1 = (xs,y) # a single input/output example

    xs = ([1,9,7],[8,8,7]) # inputs
    y = [4] # output
    ex2 = (xs,y) # another input/output example

    task = [ex1,ex2] # a task is a list of input/outputs
    
    assert fe.forward_one_task( task).shape == torch.Size([H])

    # a few more tasks
    task2 = [ex1]
    task3 = [ex2]

    # batched forward pass - test cases
    assert fe.forward([task,task2,task3]).shape == torch.Size([3,H]) 
    assert torch.all( fe.forward([task,task2,task3])[0] == fe.forward_one_task(task) )
    assert torch.all( fe.forward([task,task2,task3])[1] == fe.forward_one_task(task2) )
    assert torch.all( fe.forward([task,task2,task3])[2] == fe.forward_one_task(task3) )

    # pooling of examples happens through averages - check via this assert
    assert(torch.stack([fe.forward_one_task(task3),fe.forward_one_task(task2)],0).mean(0) - fe.forward_one_task(task)).abs().max() < 1e-5
    assert(torch.stack([fe.forward_one_task(task),fe.forward_one_task(task)],0).mean(0) - fe.forward_one_task(task3)).abs().max() > 1e-5

    from program import Program, Function, Variable, BasicPrimitive, New

    semantics = {(name): () for name in ["if","+","0" ,"1","and","lt"]}
    primitive_types = {("if"): Arrow(BOOL, Arrow(INT, Arrow(INT, INT))),
                       ("+"): Arrow(INT, Arrow(INT, INT)),
                       ("0"): INT,
                       ("1"): INT,
                       ("and"): Arrow(BOOL,Arrow(BOOL,BOOL)),
                       ("lt"): Arrow(INT,Arrow(INT,BOOL))}
    basic_addition = BasicPrimitive("+",Arrow(INT, Arrow(INT, INT)))
    basic_zero = BasicPrimitive("0",INT)
    basic_one =  BasicPrimitive("1",INT)
    basic_if = BasicPrimitive("if",Arrow(BOOL, Arrow(INT, Arrow(INT, INT))))
    basic_and = BasicPrimitive("and",Arrow(BOOL,Arrow(BOOL,BOOL)))
    basic_comparison = BasicPrimitive("lt",Arrow(INT,Arrow(INT,BOOL)))
    
    programs = [Function(basic_addition,
                         [basic_zero,basic_one]),
                basic_zero,
                basic_one]
    tasks = [task,task2,task3]
    
    template_dsl = DSL(semantics, primitive_types)
    
    template1 = template_dsl.DSL_to_Uniform_PCFG(type_request=INT,
                                                upper_bound_type_size=4, 
                                                max_program_depth=4, 
                                                min_variable_depth=4,
                                                n_gram = 1)
    template2 = PCFG("number",{"number": {basic_if: (["bool","number","number"],1.),
                                         basic_addition: (["number","number"],1.),
                                         basic_zero: ([],1.),
                                         basic_one: ([],1.)},
                              "bool": {basic_and: (["bool","bool"],1.),
                                       basic_comparison: (["number","number"],1.)}},
                     process_probabilities=False)

    import math
    # 3 different kinds of recognition models depending on whether they use the intermediate q tensor,
    # and whether they unroll the pcfg
    models = {"has Q": (RecognitionModel(fe,template_dsl=template_dsl,type_request=INT),0.),
              "has no Q (directly predict probabilities), bounded depth (unrolled)": (RecognitionModel(fe,template_pcfg=template1), 0.),
              "has no Q (directly predict probabilities), unbounded depth": (RecognitionModel(fe,template_pcfg=template2), -3*math.log(3))
    }
    
    for model_name, (model,asymptote) in models.items():
        print("training model",model_name)
        optimizer = torch.optim.Adam(model.parameters())
        for step in range(100):
            optimizer.zero_grad()

            grammars = model(tasks, log_probabilities=True)
            likelihood = sum(g.log_probability_program(g.start,p)
                             for g,p in zip(grammars, programs) )
            (-likelihood).backward()
            optimizer.step()
            print("optimization step",step,"\tlog likelihood ",likelihood)
        

        grammars = model(tasks)
        for g,p in zip(grammars, programs):
            print("predicted grammar:",g)
            print("intended program:",p)
            print("probability of intended program:",g.probability_program(g.start,p))
            samples = [g.sample_program(g.start) for _ in range(100)]
            print("100 samples:",samples)
            print("what fraction of the samples are correct?",
                  sum(str(s) == str(p) for s in samples ))
            print()

        print("asymptotically, the likelihood should converge to",asymptote,"and it actually converged to",likelihood)

    tasks = [task,task2,task3]
    
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(2000):
        optimizer.zero_grad()

        grammars = model(tasks, log_probabilities=True)

        likelihood = sum(log_probability_program(g, template.start, p)
                         for g,p in zip(grammars, programs))
        (-likelihood).backward()
        optimizer.step()
        if step % 100 == 0:
            print("optimization step",step,"\tlog likelihood ",likelihood)

    from math import exp

    def normalise(rules):
        normalised_rules = {}
        for S in rules:
            s = sum(exp(rules[S][P][1].item()) for P in rules[S])
            if s > 0:
                normalised_rules[S] = {}
                for P in rules[S]:
                    normalised_rules[S][P] = rules[S][P][0], exp(rules[S][P][1].item()) / s
        return PCFG(template.start, 
            normalised_rules, 
            max_program_depth=template.max_program_depth)        

    grammars = model(tasks)
    for g, p in zip(grammars, programs):
        grammar = normalise(g)
        print(grammar)
        print("program", p)
        print(grammar.probability_program(template.start, p))

    # asymptotically the likelihood should converge to -3ln3. it does on my machine (Kevin)
