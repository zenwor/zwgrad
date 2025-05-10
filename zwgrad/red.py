import zwgrad as zwg
import zwgrad.nn as nn
from zwgrad.zwir.op import OP


class GlobalAvgPool(nn.Module):
    def __init__(self):
        pass

    def fwd(self, x: zwg.Tensor, axis=None, keepdims=False):
        if axis is None:
            axis = (-2, -1)

        sum_ten = zwg.Tensor._make_op_tensor(
            OP.RED, [x], {"red_met": "sum", "axis": axis, "keepdims": keepdims}
        )
        red = zwg.Tensor._make_op_tensor(OP.DIV, [sum_ten, x], arg="size")
        return red


global_avg_pool = GlobalAvgPool()
