import torch
import torch.nn as nn


# [2,4,4]
a = torch.tensor(
    [[[1,2,3,4],
      [5,6,7,8],
      [9,10,11,12],
      [13,14,15,16]],
     [[-1, -2, -3, -4],
      [-5, -6, -7, -8],
      [-9, -10, -11, -12],
      [-13, -14, -15, -16]]
     ])
# print(a)
# # c = a/2
# # print(c)
#
a1 = a[:,0,:]  # [2,4]
print(a1)
a2 = a[:,1,:]
print(a2)
#
# print(a1*a2)

# y = torch.cat((a1, a2), dim=0)
# print(y)

# r = torch.tensor([[1,2,3]])
# print(r.reshape(3))
