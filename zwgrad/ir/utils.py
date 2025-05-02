IR_CNT = 1
NAME2OBJ = {}
OBJ2NAME = {}

def gen_name(template: str = None) -> str:
    global IR_CNT

    name = f"obj_{IR_CNT}"
    IR_CNT += 1

    if template:
        return f"{template}{name}"

    return name

def get_obj(name: str):
    global NAME2OBJ
    return NAME2OBJ.get(name, None)

def get_name(obj) -> str:
    global OBJ2NAME
    return OBJ2NAME.get(obj, None)
