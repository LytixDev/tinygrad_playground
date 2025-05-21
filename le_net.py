from typing import List, Callable
from tinygrad import Tensor, nn, Device, TinyJit
from tinygrad.nn.datasets import mnist


class LeNet:
  def __init__(self):
    self.conv1 = nn.Conv2d(1, 4, 5)
    self.conv2 = nn.Conv2d(4, 12, 5)
    self.fc1 = nn.Linear(12*4*4, 10)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).tanh()
    x = x.avg_pool2d()
    x = self.conv2(x).tanh()
    x = x.avg_pool2d()
    x = x.flatten(1)
    x = self.fc1(x)
    return x


print(Device.DEFAULT)
X_train, Y_train, X_test, Y_test = mnist()
model = LeNet()
optim = nn.optim.SGD(nn.state.get_parameters(model))
batch_size = 32

@TinyJit
def step():
  Tensor.training = True
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss

# about 86% acc on MNIST
for s in range(5_000):
  loss = step()
  if s % 100 == 0:
    Tensor.training = False
    acc = (model(X_test).softmax().argmax(axis=1) == Y_test).mean().item()
    print(f"step {s:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
