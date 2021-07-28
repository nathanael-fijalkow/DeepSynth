import torch
import random

from pcfg import PCFG
from cons_list import tuple2constlist
from torch import tensor
from predictions_experiments import Data

class Object_sampler():
    """
    Object to sample element of a given type together with constraints parameters. For now just type list[int]
    max_size = max number of elements in a list
    min_int/max_int = minimum/maximum integer that we allow
    """
    def __init__(self, max_size = 5, min_int = 0, max_int = 10) -> None:
        self.max_size = max_size
        self.min_int = min_int
        self.max_int = max_int

    def sample(self, type = 'list_int'):
        if type == 'list_int':
            size = random.randint(1,self.max_size)
            res = [random.randint(self.min_int,self.max_int) for _ in range(size)]
            return res
        assert(False)


class Dataset(torch.utils.data.IterableDataset):
    """
        Object that represent a dataset as an iterator. It gives a stream of pairs ([I,O], program)
        size_dataset : size of the dataset
        nb_max_IOs : maximal number of IOs in a example
        nb_variables : number of variables we need for each example
        transform : a function to transform the raw [I,O], program
        min_int, max_int : min/max integer that we allow (in particular, if an output has a too big or too small integer, we remove this example)
    """
    def __init__(self, size_dataset, dsl, pcfg, nb_max_IOs = 2, nb_variables = 2, transform = None, min_int = 0, max_int = 10):
        super(Dataset).__init__()
        self.size_dataset = size_dataset
        self.dsl = dsl
        self.pcfg = pcfg
        self.object_sampler = Object_sampler(min_int = min_int, max_int = max_int)
        self.program_sampler = pcfg.sampling()
        self.transform = transform
        self.nb_max_IOs = nb_max_IOs
        self.nb_variables = nb_variables

    def __iter__(self):
        return (self.__single_data__() for i in range(self.size_dataset))

    def __single_data__(self, verbose = False):
        flag = True
        output = None
        while flag == True or output == None:
            program = next(self.program_sampler)
            nb_IOs = random.randint(1,self.nb_max_IOs)
            inputs = [[self.object_sampler.sample() for _ in range(self.nb_variables)] for _ in range(nb_IOs)]
            try:
                outputs = []
                for input in inputs:
                    environment = tuple2constlist(input)
                    output = program.eval_naive(self.dsl, environment)
                    if self.__is_ok__(output):
                        outputs.append(output)
                    else:
                        raise ValueError()
                flag = False   
            
            except:
                pass
        
        x = [[I,O] for I,O in zip(inputs, outputs)]
        y = program
        if verbose:
            print(program)
        if self.transform:
            x = self.transform.encode_all_examples(x)
            y = self.transform.encode_program(y)
        return x, y
    
    def __is_ok__(self,tensor):
        """
        a condition to accept or reject a tensor for the dataset
        """
        if len(tensor)==0: return False
        for e in tensor:
            if not(self.object_sampler.max_int <= e and e <= self.object_sampler.max_int):
                return False
        return True


def custom_collate(batch):
    return [batch[i][0] for i in range(len(batch))], torch.stack([batch[i][1] for i in range(len(batch))])

def custom_collate_2(batch):
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]

