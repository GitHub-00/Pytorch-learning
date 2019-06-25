import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = torch.relu(x).data.numpy()
y_sigmod = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_sofplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')


plt.subplot(222)
plt.plot(x_np, y_sigmod, c='red', label='sigmod')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')


plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')


plt.subplot(224)
plt.plot(x_np, y_sofplus, c='red', label='sofplus')
plt.ylim((-0.2, 0.6))
plt.legend(loc='best')

plt.show()

