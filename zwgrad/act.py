import zwgrad as zwg
from zwgrad.nn.module import Module
from zwgrad.zwir.op import OP


class ReLU(Module):
    def __init__(self):
        pass

    def fwd(self, x: zwg.Tensor) -> zwg.Tensor:
        zs = zwg.Tensor._make_op_tensor(OP.FILL_LIKE, [x], 0.0)
        mx = zwg.Tensor._make_op_tensor(OP.MAX, [x, zs])
        return mx


relu = ReLU()
