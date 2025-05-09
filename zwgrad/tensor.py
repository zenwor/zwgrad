import numpy as np
from zwir.op import OP, OPNode


class Tensor:
    def __init__(self, data, device: str = "cpu"):
        if isinstance(data, OPNode):
            self.op = data
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.op = OPNode(OP.TEN, [data])

        self.device = device
        self._shape = data.shape if hasattr(data, "shape") else None
        self._dtype = data.dtype if hasattr(data, "dtype") else None

    @property
    def shape(self):
        return self.op.exec().shape if self._shape is None else self._shape

    @property
    def dtype(self):
        return self.op.exec().dtype if self._dtype is None else self._dtype

    def __str__(self):
        return f"Tensor(op={self.op}, shape={self.shape}, dtype={self.dtype}, device={self.device})"  # noqa: E501

    def numpy(self):
        return self.op.exec()

    # Op overloading
    def __add__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
        assert self.device == x.device, "Tensors must be on same device."
        return Tensor(OPNode(OP.ADD, [self.op, x.op]))

    # Op overloading
    def __mul__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
        assert self.device == x.device, "Tensors must be on same device."
        return Tensor(OPNode(OP.MUL, [self.op, x.op]))

    def __matmul__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
        assert self.device == x.device, "Tensors must be on same device."
        return Tensor(OPNode(OP.MATMUL, [self.op, x.op]))
