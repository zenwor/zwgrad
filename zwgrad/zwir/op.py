import numpy as np


class OP:
    # NO OPS
    TEN = "TENSOR"

    # UNARY OPS
    SUM = "SUM"
    RESHAPE = "RESHAPE"

    # BINARY OPS
    ADD = "ADD"
    MUL = "MUL"
    MATMUL = "MATMUL"


NOOPS = ["TEN"]
UNARYOPS = ["SUM", "RESHAPE"]
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
        self.owner = None

    def __str__(self):
        src_repr = [
            ("  " + str(s)) if isinstance(s, OPNode) else repr(s) for s in self.src
        ]  # noqa: E501
        src_repr = [_src.replace("\n", "") for _src in src_repr]
        src_repr = ",\n ".join(src_repr)
        if self.op != OP.TEN:
            src_repr = "\n " + src_repr
            return f"OPNode(op={self.op}, src=({src_repr}),\n arg={self.arg})"
        else:
            return f"OPNode(op={self.op}, src=({src_repr}), arg={self.arg})"

    # fmt: off
    def exec(self):
        # flake8: noqa
        if self._val is not None:
            return self._val

        a, b = None, None
        if is_unary(self.op):
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]
        if is_binary(self.op):
            a = self.src[0].exec() if isinstance(self.src[0], OPNode) else self.src[0]
            b = self.src[1].exec() if isinstance(self.src[1], OPNode) else self.src[1]

        # NO OPS
        if self.op == OP.TEN:
            assert len(self.src) == 1, "Tensor source must be of size 1."
            self._val = self.src[0]
        # UNARY OPS
        elif self.op == OP.SUM:
            self._val = np.sum(a, axis=self.arg["axis"], keepdims=self.arg["keepdims"])
        elif self.op == OP.RESHAPE:
            self._val = np.reshape(a, self.arg["shape"])
        # BINARY OPS
        elif self.op == OP.ADD:
            self._val = np.add(a, b)
        elif self.op == OP.MUL:
            self._val = np.multiply(a, b)
        elif self.op == OP.MATMUL:
            self._val = np.matmul(a, b)
        else:
            raise ValueError(f"Invalid OPNode op: {self.op}")
        return self._val
    # fmt: on
