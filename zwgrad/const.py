class OP:
    ADD = "add"
    MUL = "mul"

    def __getattr__(cls, name):
        if name.isupper():
            return name.lower()
        raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")
