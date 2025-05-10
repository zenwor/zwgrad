import numpy as np


class OP:
    # NO OPS
    TEN = "TENSOR"

    # UNARY OPS
    FILL_LIKE = "FILL_LIKE"
    SUM = "SUM"
    RESHAPE = "RESHAPE"
    RED = "REDUCE"

    # BINARY OPS
    ADD = "ADD"
    MUL = "MUL"
    DIV = "DIV"  # Scalar, for now
    MATMUL = "MATMUL"
    MAX = "MAX"


NOOPS = ["TEN"]
UNARYOPS = ["FILL_LIKE", "SUM", "RESHAPE", "REDUCE"]
BINARYOPS = ["ADD", "MUL", "DIV", "MATMUL", "MAX"]


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

    def __str__(self, indent=0):
        indent_str = " " * indent

        if len(self.src) == 1:
            s = self.src[0]
            if isinstance(s, OPNode):
                src_str = f"({s.__str__(indent + 4).lstrip()})"
            else:
                if isinstance(s, np.ndarray):
                    src_str = f"(<ndarray at {hex(id(s))})"
                else:
                    src_repr = repr(s).replace("\n", " ")
                    src_str = f"({src_repr})"
        else:
            src_repr = []
            for s in self.src:
                if isinstance(s, OPNode):
                    src_repr.append(s.__str__(indent + 4))
                else:
                    if isinstance(s, np.ndarray):
                        src_str = f"<ndarray at {hex(id(s))}>"
                    else:
                        src_str = repr(s).replace("\n", " ")
                    src_repr.append(f"{' ' * (indent + 4)}{src_str}")
            src_str = (
                f"{' ' * indent}[]"
                if not src_repr
                else "[\n" + ",\n".join(src_repr) + "\n" + f"{' ' * indent}]"
            )

        return f"{indent_str}OPNode(op={self.op}, src={src_str}, arg={self.arg})"

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
        elif self.op == OP.FILL_LIKE:
            self._val = np.empty_like(a)
            self._val.fill(self.arg)
        elif self.op == OP.RED:
            # TODO: Add support and possible better sync with other OPs
            red_met = self.arg["red_met"]
            if red_met == "sum":
                self._val = np.sum(a, axis=self.arg["axis"], keepdims=self.arg["keepdims"])
        # BINARY OPS
        elif self.op == OP.ADD:
            self._val = np.add(a, b)
        elif self.op == OP.MUL:
            self._val = np.multiply(a, b)
        elif self.op == OP.DIV:
            if self.arg == "size":
                b = b.size
            self._val = np.divide(a, b)
        elif self.op == OP.MATMUL:
            self._val = np.matmul(a, b)
        elif self.op == OP.MAX:
            self._val = np.maximum(a, b)
        else:
            raise ValueError(f"Invalid OPNode op: {self.op}")
        return self._val
    # fmt: on
