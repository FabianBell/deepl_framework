"""
Utils package
"""

class ModelOutput:
    """
    Model output wrapper
    """

    def __init__(self, **kwargs):
        self._elems = kwargs

    def __getattr__(self, name):
        return self._elems[name]
    
    def __getitem__(self, name):
        return self.__getattr__(name)

    def __iter__(self):
        for elem in self._elems:
            yield elem

def unpack(inp, tar):
    """
    Unpack the data so that it can be processed by pytorch_lightning
    distributed learning
    """
    return (*[v for v in inp.values()], tar, list(inp.keys()))

def pack(batch):
    """
    Requries the data to be in the format

     >>> (*elem, target, [*keys])

    The output will be a tuple of {keys: elem}
    and target
    """
    inp = batch[:-2]
    target = batch[-2]
    keys = batch[-1]
    inp = {k : v for k, v in zip(keys, inp)}
    return inp, target
        

