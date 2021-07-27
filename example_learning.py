import torch 

from predictions_experiments import Net
from dataset_sampler import Dataset, custom_collate
from encoding import Encoding
import dsl
from DSL.deepcoder import *


# A toy example 
deepcoder = dsl.DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
template_cfg = deepcoder.DSL_to_CFG(type_request)
deepcoder_pcfg = deepcoder.DSL_to_Random_PCFG(type_request, alpha=0.7)
E = Encoding(template_cfg, 5, 5)

ds = Dataset(100_000_000,deepcoder, deepcoder_pcfg, transform = E)
batch_size = 200
dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, collate_fn = custom_collate)

model = Net(template_cfg, E, 100)
# to use a saved model
# M = torch.load(PATH_IN)

# Optimizers
loss = torch.nn.BCELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.SGD(M.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 1
for epoch in range(EPOCHS):
    for i, (X, y) in enumerate(dl):  # batch of data
        model.zero_grad()
        output = model(X)
        loss_value = loss(output, y)
        if i % 100 == 0:
            print("optimization step", batch_size*i,
                  "\tbinary cross entropy ", float(loss_value))
        loss_value.backward()
        optimizer.step()

