# you can ignore this
try:
    import tqdm
except ModuleNotFoundError:
    class _SimpleTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc
            print("Warning: tqdm not installed.")
            print(desc)

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, desc):
            self.desc = desc
            print(desc)

    class tqdm:
        @staticmethod
        def tqdm(iterable, desc=None):
            return _SimpleTqdm(iterable, desc=desc)
