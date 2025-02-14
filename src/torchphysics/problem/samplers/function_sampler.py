
# TODO: Make this a DataLoader?
# More functionalties? Static? Sampling strats....
# Somehow compute locations in here...
class FunctionSampler:

    def __init__(self, n_functions, function_set):
        self.n_functions = n_functions
        self.function_set = function_set

    
    def sample_functions(self, locations, device="cpu"):
        return self.function_set.sample_functions(self.n_functions, locations, device=device)