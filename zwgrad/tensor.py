from ctx import GRAPH, OP
from ir.utils import NAME2OBJ, OBJ2NAME, gen_name

import zwgrad


class Tensor:
    def __init__(self, data, name = None, device = "cpu"):
        if isinstance(data, Tensor):
            self.copy_tensor(data)
        else:
            self.name = name if name else gen_name()
            self.data = data
            self.device = device
            self.shape = data if data is not None else None
            self.dtype = data.dtype if data is not None else None

            self.irnode = None

            NAME2OBJ[self.name] = self
            OBJ2NAME[self] = self.name

    def item(self):
        if self.irnode is not None:
            self.irnode()
            self.irnode = None

        return self.data

    def __str__(self):
        return f"Tensor(name={self.name}, data={self.data}, shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def __add__(self, other):
        if zwgrad.TRACING:
            out_name = gen_name(f"{OP.ADD}_")
            out_tensor = Tensor(None, name=out_name)

            op_data = {
                "op": OP.ADD,
                "operands": [self, other],
                "out": out_tensor,
            }
            op = GRAPH.make_op(op_data)

            GRAPH.add_op(op)
            out_tensor.irnode = op

            return out_tensor
        else:
            return Tensor(self.item() + other.item())
        
    def __mul__(self, other):
        if zwgrad.TRACING:
            out_name = gen_name(f"{OP.MUL}_")
            out_tensor = Tensor(None, name=out_name)

            op_data = {
                "op": OP.MUL,
                "operands": [self, other],
                "out": out_tensor,
            }
            op = GRAPH.make_op(op_data)

            GRAPH.add_op(op)
            out_tensor.irnode = op

            return out_tensor
        else:
            return Tensor(self.item() * other.item())

    def copy_tensor(self, tensor):
        self.name = tensor.name
        self.data = tensor.data
        self.device = tensor.device
        self.shape = tensor.shape
        self.dtype = tensor.dtype
