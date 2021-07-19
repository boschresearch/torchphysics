"""Example script to check how good the model
can approximate solutions with different thermal conductivity D
"""
import os

import torch
import time
from torchphysics.models import SimpleFCN

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

torch.set_grad_enabled(False)

"""
for i in range(8):
    x = torch.randn(2**(2*i), 2).to('cuda')
    t = torch.randn(2**(2*i), 1).to('cuda')
    D = torch.randn(2**(2*i), 1).to('cuda')
    input = {'x': x, 't': t, 'D': D}
    start = time.time()
    out = model.forward(input)
    end = time.time()
    print(2**(2*i), end-start)
"""
i = 127008

model = SimpleFCN(input_dim=4)
model = model.to('cuda')
x = torch.randn(i, 2)#.to('cuda')
t = torch.randn(i, 1)#.to('cuda')
D = torch.randn(i, 1)#.to('cuda')
start = time.time()
x = x.to('cuda')
t = t.to('cuda')
D = D.to('cuda')
input = {'x': x, 't': t, 'D': D}
out = model.forward(input)
end = time.time()
print(i, end-start)

# für gleiche Anzahl an 826281 Punkten (in der Zeit z.B. wahrscheinlich nicht unbedingt nötig)
# auf einer GPU / einer CPU
# Auswertung des Netzes inkl verschieben der Inputs auf GPU: ~0.00658416748046875
# FDM-Solution: ~0.08073807705659419
# 127008 Punkte:
# Auswertung des Netzes: 0.0038864612579345703
# FDM-Solution: ~0.04086041799746454
#
# laden des Modells auf GPU: etwa 4s
