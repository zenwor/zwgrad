import numpy as np


class OP:
    TEN = "TENSOR"
    ADD = "ADD"
    MUL = "MUL"
    MATMUIL = "MATMUL"


class OPNode:
    def __init__(self, op, src, arg=None):
        self.op = op
        self.src = src
        self.arg = arg
        self._val = None

    def __str__(self):
        src_repr = [
            str(s) if isinstance(s, OPNode) else repr(s) for s in self.src
        ]  # noqa: E501
        return f"OPNode(op={self.op}, src={src_repr}, arg={self.arg})"

    # fmt: off
    def exec(self):
        if self._val is not None:
            return self._val

        if self.op == OP.TEN:
            assert len(self.src) == 1, "Tensor source must be of size 1."
            self._val = self.src[0]
        elif self.op == OP.ADD:
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]  # noqa: E501
            b = self.src[1].exec() if isinstance(self.src[1], OPNode) else self.src[1]  # noqa: E501
            self._val = np.add(a, b)
        elif self.op == OP.MUL:
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]  # noqa: E501
            b = self.src[1].exec() if isinstance(self.src[1], OPNode) else self.src[1]  # noqa: E501
            self._val = np.multiply(a, b)
        elif self.op == OP.MATMUL:
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]  # noqa: E501
            b = self.src[1].exec() if isinstance(self.src[1], OPNode) else self.src[1]  # noqa: E501
            self._val = np.matmul(a, b)
        else:
            raise ValueError(f"Invalid OPNode op: {self.op}")
        return self._val
    # fmt: on
