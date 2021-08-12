"""DataCreater handle the creation of the tranings data.
They handle the connection of the data points of different domains. 
"""
import abc
import numpy as np

class DataCreator():
    """Parent class for all DataCreators. 
    Depending on the type of data, the class is split in InnerDataCreator and
    BoundaryDataCreator, if a boundary condition is used.

    Parameters
    ----------
    variables : dic
        A dictionary containg all variables of the problem.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training. 
        If an int is given, the methode will use at least as many data points as the
        int. For grid sampling the number may be increased, to create a fitting grid.
        The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers. 
    """
    def __init__(self, variables, dataset_size):
        self.variables = variables
        self.dataset_size = dataset_size

    @abc.abstractmethod
    def divide_dataset_for_int(self):
        """Handles how the dataset_size is divided if an int is given."""
        return

    @abc.abstractmethod
    def get_data(self):
        """Creates a dataset"""
        return

    def create_dataset_dic(self):
        '''Handels the divsion of the number of dataset points
        '''
        if isinstance(self.dataset_size, int):
            return self.divide_dataset_for_int()
        elif isinstance(self.dataset_size, (list, tuple)):
            i = 0
            dataset_dic = {}
            for vname in self.variables:
                dataset_dic[vname] = self.dataset_size[i]
                i = i + 1
            return dataset_dic
        elif isinstance(self.dataset_size, dict):
            return self.dataset_size
        else:
            raise TypeError(f"""Got type {type(self.dataset_size)} but expected
                             one of list, tuple, dict or int.""")

    def reorder_datapoints(self, data):
        '''Creates all possible combinations of the datapoints
        '''
        # Create all combinations w.r.t the index of the data.
        # Needed since meshgrid only works with 1D-Array inputs
        combinations = [np.arange(0, len(d)) for _, d in data.items()]
        index = np.array(np.meshgrid(*combinations)).T.reshape(-1, len(data))
        i = 0
        # Create a new dic with the new data combinations
        new_data_dic = {}
        for vname in data:
            new_data_dic[vname] = data[vname][index[:, i]]
            i = i + 1
        return new_data_dic


class InnerDataCreator(DataCreator):
    """Handels the creation of datasets if only inner points of the domains
    are needed. Used for DiffEqCondition.

    Parameters
    ----------
    variables : dic
        A dictionary containg all variables of the problem.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training. 
        If an int is given, the methode will use at least as many data points as the
        int. For grid sampling the number may be increased, to create a fitting grid.
        The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers. 
    sampling_strategy : str
        The sampling strategy used to sample data points. See domains
        for more details.
    """
    def __init__(self, variables, dataset_size, sampling_strategy):
        super().__init__(variables, dataset_size)
        self.sampling_strategy = sampling_strategy

    def reorder_datapoints(self, data):
        if self.sampling_strategy == 'random' and isinstance(self.dataset_size, int):
            return data
        return super().reorder_datapoints(data)

    def divide_dataset_for_int(self):
        dataset_dic = {}
        # If the strategy is random, sample for every variable the same 
        # amount of points and concatenate them later
        if self.sampling_strategy == 'random':
            for vname in self.variables:
                dataset_dic[vname] = self.dataset_size    
            return dataset_dic  
        # If the strategy is grid, devide the number of points depending 
        # on the dimension of each variable
        elif self.sampling_strategy == 'grid':
            sum_of_dim = 0
            for vname in self.variables:
                sum_of_dim = sum_of_dim + self.variables[vname].domain.dim
            scaled_dataset_size = int(np.ceil(self.dataset_size**(1/sum_of_dim)))
            for vname in self.variables:
                dataset_dic[vname] = scaled_dataset_size** \
                                     self.variables[vname].domain.dim
            return dataset_dic  
        else:
            raise NotImplementedError('The startegy ' + self.sampling_strategy 
                                      + 'is not implemented in InnerDataCreator')

    def get_data(self):
        data = {}
        dataset_dic = super().create_dataset_dic()
        for vname in self.variables:
            data[vname] = self.variables[vname].domain.sample_inside(
                dataset_dic[vname],
                type=self.sampling_strategy
            )
        data = self.reorder_datapoints(data)
        return data


class BoundaryDataCreator(DataCreator):
    """Handels the creation of datasets if boundary points of one domain
    are needed. Used for BoundaryCondition.

    Parameters
    ----------
    variables : dic
        A dictionary containg all variables of the problem.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training. 
        If an int is given, the methode will use at least as many data points as the
        int. For grid sampling the number may be increased, to create a fitting grid.
        The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers. 
    sampling_strategy : str
        The sampling strategy used to sample data points. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points.
        See domains for more details.
    """    
    def __init__(self, variables, dataset_size, sampling_strategy, 
                 boundary_sampling_strategy):
        super().__init__(variables, dataset_size)
        self.sampling_strategy = sampling_strategy
        self.boundary_sampling_strategy = boundary_sampling_strategy
        self.boundary_variable = None # set later 

    def get_data(self):
        data = {}
        dataset_dic = super().create_dataset_dic()
        for vname in self.variables:
            if vname == self.boundary_variable:
                data[vname] = self.variables[vname].domain.sample_boundary(
                    dataset_dic[vname],
                    type=self.boundary_sampling_strategy
                    )
            else:
                data[vname] = self.variables[vname].domain.sample_inside(
                    dataset_dic[vname],
                    type=self.sampling_strategy
                )
        data = self.reorder_datapoints(data)
        return data

    def reorder_datapoints(self, data):
        # For list, tuple or dict, just mesh all data points
        if isinstance(self.dataset_size, (list, tuple, dict)):
            return super().reorder_datapoints(data) 
        # Are all points random or the boundary domain is 1D, then the data
        # already has the right structure.
        elif (self.sampling_strategy == 'random'
              and (self.boundary_sampling_strategy == 'random'
                   or self.variables[self.boundary_variable].domain.dim == 1)):
            return data
        # If the the boundary domain is 1D and grid sampling was given,
        # we just have to mesh the data points together.
        elif (self.variables[self.boundary_variable].domain.dim == 1
              and not self.boundary_sampling_strategy == 'random'):
            return super().reorder_datapoints(data) 
        # When the inner sampling is random just mesh the boundary grid with
        # the inner points.
        elif (self.sampling_strategy == 'random' 
              and self.boundary_sampling_strategy == 'grid'):
            return self.mesh_inner_and_boundary_data(data, 
                                                     data[self.boundary_variable])
        elif (self.sampling_strategy == 'grid' 
              and self.boundary_sampling_strategy == 'random'):
            # First create a mesh of the inner data points
            inner_data_dic = data.copy()
            del inner_data_dic[self.boundary_variable]
            inner_data_dic = super().reorder_datapoints(inner_data_dic)
            # Secondly mesh the boundary and inner points
            return self.mesh_inner_and_boundary_data(inner_data_dic, 
                                                     data[self.boundary_variable])
        return super().reorder_datapoints(data)

    def divide_dataset_for_int(self):
        dataset_dic = {}
        # If the strategie is random and not grid at the boundary, sample for every
        # variable the same amount of points and concatenate them later.
        if (self.sampling_strategy == 'random' 
            and self.boundary_sampling_strategy == 'random'):
            for vname in self.variables:
                dataset_dic[vname] = self.dataset_size   
            return dataset_dic
        # If boundary is 1D, there are special strategies:
        elif self.variables[self.boundary_variable].domain.dim == 1:
            return self.divide_dataset_for_1D_boundary()
        # If we have the combination of random and grid, we take the root 
        # of the dataset_size and later concatenate the random points. Then
        # do a meshgrid with the boundary points.
        elif (self.sampling_strategy == 'random' 
              and self.boundary_sampling_strategy == 'grid'):
            scaled_dataset_size = int(np.ceil(np.sqrt(self.dataset_size)))
            for vname in self.variables:
                dataset_dic[vname] = scaled_dataset_size
            return dataset_dic
        # If the strategy is grid, devide the number of points depending 
        # on the dimension of each variable
        elif self.sampling_strategy == 'grid':
            return self.create_dataset_size_dic_for_grid()
        else:
            raise NotImplementedError('The combination ' + self.sampling_strategy
                                      + ' with ' + self.boundary_sampling_strategy
                                      + ' is not implemented in BoundaryCondition')

    def create_dataset_size_dic_for_grid(self):
        dataset_dic = {}
        sum_of_dim = 0
        # for random boundary points, we divide the points almost like
        # the case random grid. Just also divide the inner dataset size for the grid.
        scaled_dataset_size = self.dataset_size
        if self.boundary_sampling_strategy == 'random':
            sum_of_dim -= self.variables[self.boundary_variable].domain.dim
            scaled_dataset_size = int(np.ceil(np.sqrt(scaled_dataset_size)))
        for vname in self.variables:
            sum_of_dim = sum_of_dim + self.variables[vname].domain.dim
        root_dataset_size = int(np.ceil(scaled_dataset_size**(1/sum_of_dim)))
        scaled_dataset_size = root_dataset_size**sum_of_dim
        # Create the dictionary
        for vname in self.variables:
            if (vname == self.boundary_variable
                and self.boundary_sampling_strategy == 'random'):
                dataset_dic[vname] = scaled_dataset_size
            else:
                dataset_dic[vname] = root_dataset_size** \
                                     self.variables[vname].domain.dim
        return dataset_dic

    def divide_dataset_for_1D_boundary(self):
        dataset_dic = {}
        scaled_dataset_size = self.dataset_size
        # If inside sampling == grid, divide the number of points accordingly
        if self.sampling_strategy == 'grid':
            # For random bondary take again the root
            if self.boundary_sampling_strategy == 'random':
                points_for_boundary = int(np.ceil(np.sqrt(self.dataset_size)))
                scaled_dataset_size = points_for_boundary
            # If we need points at one side of the interval, it is
            # enough to get one single point and later create a meshgrid
            if (self.boundary_sampling_strategy == 'lower_bound_only'
                or self.boundary_sampling_strategy == 'upper_bound_only'):
                points_for_boundary = 1
            # Else we take two points (one at each boundary) and half the total
            # dataset size, so the meshgrid later gives the right amount of points
            elif self.boundary_sampling_strategy == 'grid':
                points_for_boundary = 2
                scaled_dataset_size = int(np.ceil(scaled_dataset_size/2))
            sum_of_dim = -1 # substract the dimension of the boundary domain
            for vname in self.variables:
                sum_of_dim = sum_of_dim + self.variables[vname].domain.dim
            scaled_dataset_size = int(np.ceil(scaled_dataset_size**(1/sum_of_dim)))    

        # Create the dictionary for the dataset sizes
        for vname in self.variables:
            if self.sampling_strategy == 'random':
                dataset_dic[vname] = self.dataset_size
            else:        
                if vname == self.boundary_variable:
                    dataset_dic[vname] = points_for_boundary
                else:
                    dataset_dic[vname] = scaled_dataset_size** \
                                         self.variables[vname].domain.dim
        return dataset_dic

    def mesh_inner_and_boundary_data(self, inner_data_dic, bound_data):
        num_of_elements = np.arange(0, len(bound_data))
        index = np.array(np.meshgrid(num_of_elements, num_of_elements)).T.reshape(-1,2)
        # Create a new dic with the new data combinations
        new_data_dic = {}
        for vname in self.variables:
            if vname == self.boundary_variable:
                new_data_dic[vname] = bound_data[index[:, 0]]
            else:
                new_data_dic[vname] = inner_data_dic[vname][index[:, 1]]
        return new_data_dic
