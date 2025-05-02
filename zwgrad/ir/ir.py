from ctx import OP

class IR():
    def __init__(self):
        self.graph = []

    def __str__(self):
        return str([str(irnode) for irnode in self.graph])

    def add_op(self, op):
        if isinstance(op, dict):
            op = self.make_op(op)
        self.graph.append(op)

    def make_op(self, op_data):
        return IRNode(op_data)

    def exec(self):
        # Execute node
        for node in self.graph():
            node()

class IRNode():
    def __init__(self, op_data: dict = None):
        self.op = op_data["op"] if op_data else None
        self.operands = op_data["operands"] if op_data else None
        self.out = op_data["out"] if op_data else None

    def __str__(self):
        return f"IRNode(op={self.op}, operands={[operand for operand in self.operands]}, out={self.out})"

    def __call__(self):
        return self._exec()

    def _exec(self) -> None:
        if self.op == OP.ADD:
            a = self.operands[0]
            b = self.operands[1]
            res = a + b

            self.out.copy_tensor(res)
        else:
            raise ValueError(f"Invalid IRNode op: {self.op}")
