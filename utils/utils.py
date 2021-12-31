import torch

class MaxFuncMux(object):
    def __init__(self):
        self.io = dict()
        self.io_torch = dict()
        self.torch = False

    def store(self, func, func_val):
        self.io[func] = func_val
        if type(func_val) is torch.Tensor:
            self.torch = True
            self.io_torch[func] = func_val
    
    def get_max(self):
        """
        return the max value, and the function that gives out that max value
        return func_of_max_val, max_val
        """
        if self.torch:
            key = max(self.io, key = self.io.get)
            return key, self.io_torch[key]
        else:
            return max(self.io, key = self.io.get), max(self.io.values())

class MinFuncMux(object):
    def __init__(self):
        self.io = dict()
        self.io_torch = dict()
        self.torch = False

    def store(self, func, func_val):
        self.io[func] = func_val
        if type(func_val) is torch.Tensor:
            self.torch = True
            self.io_torch[func] = func_val
    
    def get_min(self):
        """
        return the min value, and the function that gives out that min value
        return func_of_min_val, min_val
        """
        if self.torch:
            key = min(self.io, key = self.io.get)
            return key, self.io_torch[key]
        else:
            return min(self.io, key = self.io.get), min(self.io.values())