import torch 
import matplotlib.pyplot as plt

from predictions_experiments import Net
from dataset_sampler import Dataset, custom_collate, custom_collate_2
from encoding import Encoding
import dsl
from DSL.deepcoder import *


# A toy example 
min_int = -20
max_int = 20
deepcoder = dsl.DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))
template_cfg = deepcoder.DSL_to_CFG(type_request)
deepcoder_pcfg = deepcoder.DSL_to_Random_PCFG(type_request, alpha=0.7)
E = Encoding(template_cfg, 5, 5)

ds = Dataset(10_000,deepcoder, deepcoder_pcfg, transform = E, min_int= min_int, max_int = max_int)
batch_size = 30
dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, collate_fn = custom_collate)

model = Net(template_cfg, E, 100, min_int=min_int, max_int=max_int)
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
            print("parameters: \n\n", model.embed.weight,"\n\n\n")
            print("optimization step", batch_size*i,
                  "\tbinary cross entropy ", float(loss_value))
        loss_value.backward()
        optimizer.step()

# sampling = deepcoder_pcfg.sampling()
# for i in range(10):
#     print("index", i)
#     x, y = ds.__single_data__(verbose = True)
#     print(x)
#     for a,b in zip(y,model([x])[0]):
#         if a.item() == 1:
#             print(a,b)
#     print(x)
# print(len(y))

print(model.embed.weight)
print([x for x in model.embed.weight[:,0]])
x = [x for x in model.embed.weight[:,0]]
y = [x for x in model.embed.weight[:,1]]
label = [str(a+min_int) for a in range(len(x))]
plt.plot(x,y, 'o')
for i, s in enumerate(label):
    xx = x[i]
    yy = y[i]
    plt.annotate(s, (xx, yy), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()