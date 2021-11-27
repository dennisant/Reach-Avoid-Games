class MaxFuncMux(object):
    def __init__(self):
        self.io = dict()

    def store(self, func, func_val):
        self.io[func] = func_val
    
    def get_max(self):
        """
        return the max value, and the function that gives out that max value
        return func_of_max_val, max_val
        """
        return max(self.io, key = self.io.get), max(self.io.values())

class MinFuncMux(object):
    def __init__(self):
        self.io = dict()

    def store(self, func, func_val):
        self.io[func] = func_val
    
    def get_min(self):
        """
        return the min value, and the function that gives out that min value
        return func_of_min_val, min_val
        """
        return min(self.io, key = self.io.get), min(self.io.values())