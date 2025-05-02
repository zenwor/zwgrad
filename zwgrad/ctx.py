import os
from const import OP
from zwgrad.ir import IR


def str2bool(val: str) -> bool:
    return val.lower() in ("true", "1")

# Tracing is enabled by default
TRACING = str2bool(os.getenv("TRACING", "False"))
GRAPH = IR()
OP = OP() # Operation names
