# zwgrad

I decided to implement my own ML framework!

The goals are simple:
- Make it train some simple models, such as ResNet
- CUDA support
- CUDA codegen, with kernel fusions detected from IR
- MLIR, in the long run

# Changelog

## (10.05.2025) `zwgrad` supports standard operations and Linear (`zwg.nn.Lin`) layer
`zwgrad` supports basic set of operations necessary to have a standard Linear layer, with (default) or without a bias:

```py
import zwgrad as zwg
import zwgrad.nn as nn

from loguru import logger

class Model(nn.Module):
    def __init__(self):
        self.lin1 = nn.Lin(4, 8)
        self.lin2 = nn.Lin(8, 16, bias=False)

    def fwd(self, x):  # Forward pass is defined within `fwd` method
        x = self.lin1(x)
        x = self.lin2(x)
        return x

x = zwg.ten([1.0, 2.0, 3.0, 4.0], req_grad=True) # Create a tensor via `zwg.ten` & track gradient
model = Model()  # Instantiate the model
y = model(x)  # Forward pass

logger.info(f"Graph: {y}")  # Get the computation graph
logger.info(f"y = {y.numpy()}")  # Get the exact forward pass value

z = y.sum()
z.bwd()  # Backpropagate
logger.info(f"x.grad = {x.grad}")
logger.info(f"y.grad = {y.grad}")
```
