import random
import logging 
import torch
from torch import tensor

from pcfg import PCFG
from type_system import Type, PolymorphicType, PrimitiveType, Arrow, List, UnknownType, INT, BOOL
from cons_list import tuple2constlist

class Dataset(torch.utils.data.IterableDataset):
    """
        Dataset as an iterator: gives a stream of tasks
        a task is (IOs, program)

        size: size of the dataset

        nb_inputs_max: number of IOs in a task
        arguments: list of arguments for the program

        size_max: maximum number of elements in a list
        lexicon: possible values in a list
    """
    def __init__(self, 
        size, 
        dsl, 
        pcfg, 
        nb_inputs_max, 
        arguments,
        # IOEncoder,
        # IOEmbedder,
        ProgramEncoder,
        size_max,
        lexicon,
        ):
        super(Dataset).__init__()
        self.size = size
        self.dsl = dsl
        self.pcfg = pcfg
        self.input_sampler = Input_sampler(size_max = size_max, lexicon = lexicon)
        self.program_sampler = pcfg.sampling()
        self.nb_inputs_max = nb_inputs_max
        self.arguments = arguments
        # self.IOEncoder = IOEncoder
        # self.IOEmbedder = IOEmbedder
        self.ProgramEncoder = ProgramEncoder
        self.lexicon = lexicon

    def __iter__(self):
        return (self.__single_data__() for i in range(self.size))

    def __single_data__(self):
        flag = True
        output = None
        while flag or output == None:
            program = next(self.program_sampler)
            nb_IOs = random.randint(1,self.nb_inputs_max)
            inputs = [[self.input_sampler.sample(type_) for type_ in self.arguments] for _ in range(nb_IOs)]
            try:
                outputs = []
                for input_ in inputs:
                    environment = tuple2constlist(input_)
                    output = program.eval_naive(self.dsl, environment)
                    if self.__output_validation__(output):
                        outputs.append(output)
                    else:
                        raise ValueError()
                flag = False
            except:
                pass
        
        IOs = [[I,O] for I,O in zip(inputs, outputs)]
        logging.debug('Found a program:\n{}\nand inputs:\n{}'.format(program,IOs))
        return IOs, self.ProgramEncoder(program)
        # return self.IOEncoder.encode_IOs(IOs), self.ProgramEncoder(program)
    
    def __output_validation__(self, output):
        if len(output) == 0: 
            return False
        for e in output:
            if e not in self.lexicon:
                return False
        return True

class Input_sampler():
    """
    Object to sample element of a given type together with constraints parameters 
    For now we can only sample from the type list[int]

    size_max: max number of elements in an input 
    lexicon: admissible elements in a list
    """
    def __init__(self, size_max, lexicon) -> None:
        self.size_max = size_max
        self.lexicon = lexicon

    def sample(self, type):
        if type.__eq__(List(INT)):
            size = random.randint(1,self.size_max)
            res = [random.choice(self.lexicon) for _ in range(size)]
            return res
        assert(False)

