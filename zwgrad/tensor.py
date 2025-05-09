import numpy as np
from zwir.op import OP, OPNode


class Tensor:
    def __init__(self, data, device: str = "cpu", req_grad: bool = False):
        if isinstance(data, OPNode):
            self.op = data
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.op = OPNode(OP.TEN, [data])

        self.device = device
        self._shape = data.shape if hasattr(data, "shape") else None
        self._dtype = data.dtype if hasattr(data, "dtype") else None

        self.req_grad = req_grad
        self.grad = np.zeros_like(self.numpy()) if req_grad else None
        self._ctx = None  # For tracking parent tensors in the graph

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

    def _check_devices(self, x):
        return self.device == x.device

    def _binary_op_check(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
        assert self._check_devices(x), "Tensors must be on same device."

    # Op overloading
    def __add__(self, x):
        x = self.binary_op_check(x)
        res = Tensor(OPNode(OP.ADD, [self.op, x.op]))
        res._set_grad(self, x)
        return res

    def __mul__(self, x):
        x = self.binary_op_check(x)
        res = Tensor(OPNode(OP.MUL, [self.op, x.op]))
        res._set_grad(self, x)
        return res

    def __matmul__(self, x):
        x = self.binary_op_check(x)
        res = Tensor(OPNode(OP.MATMUL, [self.op, x.op]))
        res._set_grad(self, x)
        return res

    def sum(self, axis=None, keepdims=False):
        res = Tensor(
            OPNode(OP.SUM, [self.op], {"axis": axis, "keepdims": keepdims})
        )  # noqa: E501
        res._set_grad(self)
        return res

    def backward(self, grad_out=None):
        return self.bwd(grad_out)

    # Backward pass
    def bwd(self, grad_out=None):
        if not self.req_grad:
            return

        grad_out = np.ones_like(self.numpy()) if grad_out is None else grad_out
        self.grad = (
            np.zeros_like(self.numpy()) if self.grad is None else self.grad
        )  # noqa: E501
        self.grad += grad_out

        # Leaf node
        if self._ctx is None:
            return

        # UNARY OPS
        if self.op.op == OP.SUM:
            (x,) = self._ctx
            if self.op.arg["axis"] is None:
                grad = np.full_like(x.numpy(), grad_out)
            else:
                shape = list(x.shape)
                shape[self.op.arg["axis"]] = 1
                grad = np.reshape(grad_out, shape)
            x.bwd(grad)

        # BINARY OPS
        elif len(self._ctx) == 2:
            a, b = self._ctx
            if self.op.op == OP.ADD:
                a.bwd(grad_out)
                b.bwd(grad_out)
            elif self.op.op == OP.MUL:
                a.bwd(grad_out * b.numpy())
                b.bwd(grad_out * a.numpy())
            elif self.op.op == OP.MATMUL:
                a.bwd(np.matmul(grad_out, b.numpy().T))
                b.bwd(np.matmul(a.numpy().T, grad_out))

    def _set_grad(self, a, b=None):
        self.req_grad = a.req_grad or (b is not None and b.req_grad)
        if self.req_grad:
            self._ctx = (a,) if b is None else (a, b)
