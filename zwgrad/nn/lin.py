import numpy as np

import zwgrad as zwg
from zwgrad.nn.module import Module


class Lin(Module):
    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = True, device: str = "cpu"
    ):
        # w = np.random.randn(in_dim, out_dim) * 0.1
        w = np.ones((in_dim, out_dim))
        b = np.random.randn(out_dim) if bias else np.zeros(out_dim)

        self.w = zwg.ten(w, device, req_grad=True)
        self.b = zwg.ten(b, device, req_grad=True)

    def fwd(self, x: zwg.Tensor):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert (
            x.shape[1] == self.w.shape[0]
        ), f"Shape mismatch: {x.shape[1]} != {self.w.shape[0]}"

        return x @ self.w + self.b
