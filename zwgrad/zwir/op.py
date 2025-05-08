import numpy as np


class OP:
    TEN = "TENSOR"
    ADD = "ADD"
    MUL = "MUL"

    def __getattr__(cls, name):
        if name.isupper():
            return name.lower()
        raise AttributeError(
            f"'{cls.__name__}' object has no attribute '{name}'"
        )  # noqa: E501


class OPNode:
    def __init__(self, op, src, arg):
        self.op, self.src, self.arg = op, src, arg

    def __str__(self):
        return f"OPNode(op={self.op}), src=({[_src.__repr__() for _src in self.src]}), arg=({self.arg})"  # noqa: E501

    def exec(self) -> None:
        if self.op == OP.TEN:
            assert len(self.src) == 1, "Tensor source must be of size 1."
            return self.src
        if self.op == OP.ADD:
            a = self.src[0].arg
            b = self.src[1].arg
            res = self.src[2]

            res.op = OPNode(OP.TEN, res, np.add(a, b))
        else:
            raise ValueError(f"Invalid IRNode op: {self.op}")
