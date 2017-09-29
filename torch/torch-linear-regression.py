import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

N = 64

alpha = 1.3
beta = np.array([[0.5]])
 
x_data = np.random.randn(N, 1)
y_data = x_data.dot(beta) + alpha
data = np.loadtxt('../data/aerogerador.dat')
x_data = data[:,[0]]
y_data = data[:,[1]]
print(x_data.shape)
print(y_data.shape)
x = Variable(torch.Tensor(x_data), requires_grad=False)
y = Variable(torch.Tensor(y_data), requires_grad=False)

w_beta = Variable(torch.randn(1, 1), requires_grad=True)
w_alpha = Variable(torch.randn(1), requires_grad=True)

learning_rate = 1e-2
optimizer = torch.optim.Adam([w_beta, w_alpha], lr=learning_rate)

for t in range(500):
    y_pred = x.mm(w_beta).add(w_alpha.expand(2250))
    
    print(y_pred.size())
    loss = (y_pred - y).pow(2).sum()
    
    if t % 10 == 0:
        print(t, loss.data[0])
    
    optimizer.zero_grad()   
    
    loss.backward()
    
    optimizer.step()

plt.plot(x_data, )
plt.show()




