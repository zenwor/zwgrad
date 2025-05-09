import numpy as np
from zwir.op import OP, OPNode


def tensor(data, device: str = "cpu", req_grad: bool = False):
    return Tensor(data, device=device, req_grad=req_grad)


ten = tensor  # Abbreviation


class Tensor:
    def __init__(self, data, device: str = "cpu", req_grad: bool = False):
        if isinstance(data, OPNode):
            self.op = data
            self.op.owner = self  # Set owner reference
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.op = OPNode(OP.TEN, [data])
            self.op.owner = self  # Set owner reference

        self.device = device
        self._shape = data.shape if hasattr(data, "shape") else None
        self._dtype = data.dtype if hasattr(data, "dtype") else None
        self.req_grad = req_grad
        # Initialize grad only if req_grad is True
        self.grad = np.zeros_like(self.numpy()) if req_grad else None
        self._ctx = None

    @property
    def shape(self):
        return self.op.exec().shape if self._shape is None else self._shape

    @property
    def dtype(self):
        return self.op.exec().dtype if self._dtype is None else self._dtype

    def __str__(self):
        return str(self.op)

    def numpy(self):
        return self.op.exec()

    def _check_devices(self, x):
        return self.device == x.device

    def _binary_op_check(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
        assert self._check_devices(x), "Tensors must be on same device."
        return x

    @staticmethod
    def _make_op_tensor(op: OP, src=None, arg=None):
        # Ensure src contains Tensor objects
        src_tensors = [t for t in src if isinstance(t, Tensor)]

        # Create the operation node using the underlying OPNodes
        srcops = [t.op for t in src_tensors]
        res = Tensor(OPNode(op, srcops, arg))

        # Set up gradient tracking using the actual source Tensors
        if len(src_tensors) == 1:
            res._set_grad(src_tensors[0])
        elif len(src_tensors) == 2:
            res._set_grad(src_tensors[0], src_tensors[1])

        return res

    # Operations
    def __add__(self, x):
        x = self._binary_op_check(x)
        res = Tensor._make_op_tensor(OP.ADD, [self, x])
        return res

    def __mul__(self, x):
        x = self._binary_op_check(x)
        res = Tensor._make_op_tensor(OP.MUL, [self, x])
        return res

    def __matmul__(self, x):
        x = self._binary_op_check(x)
        res = Tensor._make_op_tensor(OP.MATMUL, [self, x])
        return res

    def sum(self, axis=None, keepdims=False):
        res = Tensor._make_op_tensor(
            OP.SUM, [self], {"axis": axis, "keepdims": keepdims}
        )
        return res

    def reshape(self, *shape):
        res = Tensor._make_op_tensor(OP.RESHAPE, [self], {"shape": shape})
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
        # Reshape grad_out to match self.grad's shape
        if grad_out.shape != self.grad.shape:
            grad_out = np.reshape(grad_out, self.grad.shape)
        self.grad += grad_out

        # print(f"Computing backward for op: {self.op.op}, ctx: {self._ctx}")

        # Leaf node
        if self._ctx is None:
            # print("Leaf node, stopping backward pass")
            return

        # UNARY OPS
        if self.is_unary():
            (a,) = self._ctx
            if self.op.op == OP.SUM:
                if self.op.arg["axis"] is None:
                    grad = np.full_like(a.numpy(), grad_out)
                else:
                    # Broadcast grad_out to match input shape
                    grad = np.broadcast_to(
                        np.expand_dims(grad_out, axis=self.op.arg["axis"]),
                        a.shape,  # noqa: E501
                    )
                a.bwd(grad)
            elif self.op.op == OP.RESHAPE:
                a.bwd(np.reshape(grad_out, a.shape))
        # BINARY OPS
        elif self.is_binary():
            a, b = self._ctx
            print(f"Binary op {self.op.op}, a: {a}, b: {b}")
            if self.op.op == OP.ADD:
                a.bwd(grad_out)
                b.bwd(grad_out)
            elif self.op.op == OP.MUL:
                a.bwd(grad_out * b.numpy())
                b.bwd(grad_out * a.numpy())
            elif self.op.op == OP.MATMUL:
                a.bwd(grad_out @ b.numpy().T)
                b.bwd(a.numpy().T @ grad_out)

    def _set_grad(self, a, b=None):
        """Setup gradient tracking for this operation"""
        self.req_grad = a.req_grad or (b is not None and b.req_grad)
        if self.req_grad:
            self._ctx = (a,) if b is None else (a, b)

    def is_unary(self):
        return self._ctx is not None and len(self._ctx) == 1

    def is_binary(self):
        return self._ctx is not None and len(self._ctx) == 2
