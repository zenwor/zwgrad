import numpy as np
from zwir.op import OP, OPNode


class Tensor:
    def __init__(self, data, device: str = "cpu"):
        if not isinstance(data, OPNode):
            if not isinstance(data, np.ndarray):  # Wrap into np.ndarray
                data = np.array(data)
            data = OPNode(OP.TEN, src=self, arg=data)  # Make OP holding value

        self.op = data
        self.device = device
        self.shape = data.shape if "shape" in data.__dir__() else None
        self.dtype = data.dtype if "dtype" in data.__dir__() else None

    def __str__(self):
        return f"Tensor(op={self.op}, shape={self.shape}, dtype={self.dtype}, device={self.device})"  # noqa: E501

    def _constr(self, x, op: OP):
        assert self.device == x.device, "Tensors must be on same device."

        res = Tensor(None, self.device)
        opnode = OPNode(op, (self.op, x.op, res), None)
        res.op = opnode
        return res

    def item(self):
        self._exec()  # TODO: Realize
        return self.op.arg

    def _exec(self):
        return self.op.exec()

    # Op overloads
    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __add__(self, x):
        return self._constr(x, OP.ADD)

    def __mul__(self, x):
        return self._constr(x, OP.MUL)
