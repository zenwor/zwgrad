import numpy as np


class OP:
    TEN = "TENSOR"
    ADD = "ADD"
    MUL = "MUL"
    MATMUL = "MATMUL"
    SUM = "SUM"


NOOPS = ["TEN"]
UNARYOPS = ["SUM"]
BINARYOPS = ["ADD", "MUL", "MATMUL"]


def is_unary(x: OP):
    return x in UNARYOPS


def is_binary(x: OP):
    return x in BINARYOPS


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
        # flake8: noqa
        if self._val is not None:
            return self._val

        if is_unary(self.op):
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]
        if is_binary(self.op):
            b = self.src[1].exec() if isinstance(self.src[1], OPNode) else self.src[1]

        if self.op == OP.TEN:
            assert len(self.src) == 1, "Tensor source must be of size 1."
            self._val = self.src[0]
        elif self.op == OP.ADD:
            self._val = np.add(a, b)
        elif self.op == OP.MUL:
            self._val = np.multiply(a, b)
        elif self.op == OP.MATMUL:
            self._val = np.matmul(a, b)
        elif self.op == OP.SUM:
            self._val = np.sum(a, axis=self.arg["axis"], keepdims=self.arg["keepdims"])
        else:
            raise ValueError(f"Invalid OPNode op: {self.op}")
        return self._val
    # fmt: on
