
from .functionset import FunctionSet
from ...spaces import Points


class DataFunctionSet(FunctionSet):
    def __init__(self, function_space, data):
        super().__init__(function_space, len(data))
        self.data = data
    
    @property
    def is_discretized(self):
        return True
    
    def create_functions(self, device="cpu"):
        self.data = self.data.to(device)

    def get_function(self, idx):
        return Points(self.data[idx], self.function_space.output_space)


# class DataFunctionSetDeepONet(FunctionSet):
#     # TODO: Improve this, maybe more genral inputs, …
#     def __init__(self, solution_space : FunctionSpace, branch_space : FunctionSpace, 
#                  trunk_data : torch.Tensor, branch_data : torch.Tensor, 
#                  solution_data = None,
#                  branch_discretization_data = None,
#                  parameter_data = None, parameter_space=None):
#         super().__init__(solution_space)
#         #TODO: Check that shapes are compatible and handle different shapes
#         # (when trunk_data is the same for all data etc...)
#         self.branch_space = branch_space
#         self.branch_data = Points(branch_data, branch_space.output_space)
#         self.trunk_data = Points(trunk_data, solution_space.input_space)

#         self.trunk_sampler = DataSampler(self.trunk_data)
        
#         if not branch_discretization_data is None:
#             self.branch_input_sampler = DataSampler(
#                 Points(branch_discretization_data, branch_space.input_space)
#             )
#         else: # create dummy
#             # TODO: Maybe linspace instead of empty?
#             self.branch_input_sampler = DataSampler(
#                 Points(torch.empty(1), branch_space.input_space)
#             )


#         if solution_data is None and parameter_data is None:
#             raise ValueError("Either the expected network output or other parameters need to be specified.")      
        
#         if solution_data is None:
#             self.parameter_data = Points(parameter_data, parameter_space)
#             self.data_create_fn = self._return_parameter_data
#         elif parameter_data is None: 
#             self.solution_data = Points(solution_data, solution_space.output_space)
#             self.data_create_fn = self._return_solution_data
#         else:
#             self.solution_data = Points(solution_data, solution_space.output_space)
#             self.parameter_data = Points(parameter_data, parameter_space)
#             self.data_create_fn = self._return_solution_and_parameter_data


#     def sample_functions(self, n_samples, locations, device="cpu"):
#         # TODO: Add some batching functionality, e.g. count the number 
#         # of n_samples and return data accordingly...
#         if isinstance(locations, Points):
#             return self._sample_for_single_loc(n_samples, locations, device)
        
#         outputs = []
#         for loc in locations:
#             outputs.append(self._sample_for_single_loc(n_samples, loc, device))
#         return outputs

#     def _sample_for_single_loc(self, n_samples, loc, device):
#         if loc[self.branch_space.input_space] is self.branch_input_sampler.points:
#             self.branch_data.to(device)
#             return self.branch_data[:n_samples]
        
#         return self.data_create_fn(n_samples, device)


#     def _return_solution_data(self, n_samples, device):
#         self.solution_data.to(device)
#         return self.solution_data[:n_samples]

#     def _return_parameter_data(self, n_samples, device):
#         self.parameter_data.to(device)
#         return self.parameter_data[:n_samples]

#     def _return_solution_and_parameter_data(self, n_samples, device):
#         solution_data = self._return_solution_data(n_samples, device)
#         param_data = self._return_parameter_data(n_samples, device)
#         return solution_data.joined(param_data)



# class DataFunctionSetNeuralOperators(FunctionSet):
#     """
#     Functionset for Neural operator class (Data would have different shape then for DeepONet?)
#     TODO: Implement...
#     """
#     def __init__(self, function_space):
#         super().__init__(function_space)