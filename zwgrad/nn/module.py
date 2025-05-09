class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)
