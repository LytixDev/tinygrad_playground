from typing import List, Callable
from tinygrad import Tensor, nn, Device, TinyJit
from tinygrad.nn.datasets import mnist

# print(Device.DEFAULT)


class MLP_Mnist:
  def __init__(self):
    self.l1 = nn.Linear(28 * 28, 256)
    self.l2 = nn.Linear(256, 64)
    self.l3 = nn.Linear(64, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x.flatten(1)).relu()
    x = self.l2(x).relu()
    return self.l3(x.dropout(0.5))


class CNN_Mnist:
  def __init__(self):
    # Input is 28*28*1
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3)) # After pooling we get 13*13*32
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3)) # Afte pooling we get 5*5*64
    self.l3 = nn.Linear(1600, 10) # 5*5*64 = 1600

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2), stride=2)
    x = self.l2(x).relu().max_pool2d((2,2), stride=2)
    return self.l3(x.flatten(1).dropout(0.5))
  

# Based on tinygrad/examples/beautiful_mnist.py
class Beautiful_Mnist:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)


X_train, Y_train, X_test, Y_test = mnist()

model = Beautiful_Mnist()
#optim = nn.optim.SGD(nn.state.get_parameters(model))
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128

@TinyJit
def step():
  Tensor.training = True  # makes dropout work
  # No notion of epoch, just do a bunch of mini batches
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).softmax().sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss

for s in range(1000):
  loss = step()
  if s % 100 == 0:
    Tensor.training = False
    acc = (model(X_test).softmax().argmax(axis=1) == Y_test).mean().item()
    print(f"step {s:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")