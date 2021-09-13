"""DataCreater creates training data.
Handles the connection of the data points of different domains and strategies. 
"""
import abc
import numpy as np

class DataCreator():
    """Parent class for all DataCreators. 
    Depending on the type of data, the class is split in InnerDataCreator and
    BoundaryDataCreator, if a boundary condition is used.

    Parameters
    ----------
    variables : dict
        A dictionary containg all variables of the problem.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training. 
        If an int is given, the methode will use at least as many data points as the
        int. For grid sampling the number may be increased, to create a fitting grid.
        The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers. 
    sampling_strategy : str, list or dict
        The sampling strategy used to sample data points. Either a string that 
        sets the strategy for all variables or a dic giving a specific 
        strategy per variable. See domains for more details of useable strats.
    sampling_params : dict 
        A dictionary containing additional parameters for the sampling.
        For each variable thats needs paramters the item of the dictionary needs
        to be a dictionary itself.
        E.g. if for the variable 'x' a normal distribution should be used. Then
        sampling_params = {'x': {'mean':...., 'cov':....}}.
    """
    def __init__(self, variables, dataset_size, sampling_strategy,
                 sampling_params):
        self.variables = variables
        self.dataset_size = dataset_size
        self.sampling_strategy = sampling_strategy
        self.sampling_params = sampling_params
        # Following list containing the different implemented strategies
        # and of what 'type' they are:
        # random_like = samplings without a 'fixed structure'
        # grid_like = samplings that always create the same points 
        self.random_like = ['random', 'normal']
        self.grid_like = ['grid', 'spaced_grid', 'lhs',
                          'lower_bound_only', 'upper_bound_only']

    @abc.abstractmethod
    def get_data(self):
        """Creates a dataset"""
        return

    def _create_sampling_strat_dic(self):
        if isinstance(self.sampling_strategy, str):
            self._check_strat_exists(self.sampling_strategy)
            sampling_strat_dic = {}
            for vname in self.variables:
                sampling_strat_dic[vname] = self.sampling_strategy
            return sampling_strat_dic
        if isinstance(self.sampling_strategy, list):
            sampling_strat_dic = {}
            i = 0
            for vname in self.variables:
                strat = self.sampling_strategy[i]
                self._check_strat_exists(strat)
                sampling_strat_dic[vname] = strat
                i += 1
            return sampling_strat_dic
        elif isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        else:
            raise TypeError(f"""Sampling strategy has to be a string, list or a
                            dictionary, got {type(self.sampling_strategy)}.""") 

    def _check_strat_exists(self, strat):
        in_rand = strat in self.random_like
        in_grid = strat in self.grid_like
        if not in_grid and not in_rand:
            raise NotImplementedError(f"""The strategy {strat} does not exist!""")

    def _create_dataset_dic(self):
        '''Handels the divsion of the number of dataset points.

        Returns
        -------
        dict
            The number of points for each variable in a dictionary. 
        '''
        dataset_dic, all_random = self._transform_dataset_input_to_dic() 
        # for random like sampling we want new points for each combination.
        # In this case is it best to sample random points once for all
        # point combinations. Therefore set the dataset size of these
        # variables to the product of all dataset sizes. 
        # If a list or dict is given we should always do this even if all
        # strats are random.
        if not all_random:
            self.size_prod = np.prod(list(dataset_dic.values()))
            for vname in self.variables:
                if self.sampling_strategy[vname] in self.random_like:
                    dataset_dic[vname] = self.size_prod
        return dataset_dic 

    def _transform_dataset_input_to_dic(self):
        """Takes the original input of the 'dataset_size' and transforms it to 
        a dict."""
        if isinstance(self.dataset_size, int):
            return self._divide_dataset_for_int()
        elif isinstance(self.dataset_size, (list, tuple)):
            i = 0
            dataset_dic = {}
            for vname in self.variables:
                dataset_dic[vname] = self.dataset_size[i]
                i = i + 1
            return dataset_dic, False
        elif isinstance(self.dataset_size, dict):
            return self.dataset_size, False
        else:
            raise TypeError(f"""Got type {type(self.dataset_size)} but expected
                             one of list, tuple, dict or int.""")

    def _divide_dataset_for_int(self):
        """Splits the dataset size to all variables, depending
        on the dimension of the variable. 
        """
        if self._all_strats_random():
            return self._dataset_for_all_random(), True
        dataset_dic = {}
        sum_of_dim = self._compute_sum_of_dims()
        dataset_root = self.dataset_size**(1/sum_of_dim)
        dataset_root = int(np.ceil(dataset_root))
        for vname in self.variables:
            dim = self.variables[vname].domain.dim
            dataset_dic[vname] = dataset_root**dim
        return dataset_dic, False

    def _all_strats_random(self):
        strat_list = list(self.sampling_strategy.values())
        return all(np.in1d(strat_list, self.random_like))

    def _dataset_for_all_random(self):
        dataset_dic = {}
        self.size_prod = self.dataset_size
        for vname in self.variables:
            dataset_dic[vname] = self.dataset_size
        return dataset_dic        

    def _compute_sum_of_dims(self):
        sum_of_dim = 0
        for vname in self.variables:
            sum_of_dim += self.variables[vname].domain.dim
        return sum_of_dim

    def _reorder_datapoints(self, data):
        '''Creates all possible combinations of the datapoints. 
        '''
        # First find all variables that use a grid like strat:
        grid_var = {}
        for vname in self.variables:
            if self.sampling_strategy[vname] in self.grid_like:
                grid_var[vname] = len(data[vname])
        if len(grid_var) > 1:
            data = self._create_mesh_of_grid_samples(data, grid_var)
        if len(grid_var) >= 1:
            data = self._copy_grid_to_full_length(data, grid_var)
        return data

    def _create_mesh_of_grid_samples(self, data, grid_var):
        # Create all combinations w.r.t the index of the data.
        # Need to work with the index, since meshgrid only can handle
        # 1D-Arrays. Only needed if more then one grid is sampled
        combinations = [np.arange(0, data_len) for _, data_len in grid_var.items()]
        index = np.array(np.meshgrid(*combinations)).T.reshape(-1, len(grid_var))
        i = 0
        # Create a new dic with the new data combinations
        for vname in grid_var:
            data[vname] = data[vname][index[:, i]]
            i = i + 1
        return data

    def _copy_grid_to_full_length(self, data, grid_var):
        # At last we have to combine the grid points with the random points.
        # After the construction in the methode .create_dataset_dic() the
        # random like points will have the correct size. Only the grid like
        # points now need to be copied. 
        for vname in grid_var:
            copy = self.size_prod/len(data[vname])
            data[vname] = np.tile(data[vname], (int(copy), 1))
        return data


class InnerDataCreator(DataCreator):
    """Handels the creation of datasets if only inner points of the domains
    are needed. Used for DiffEqCondition.

    Parameters
    ----------
    variables : dict
        A dictionary containg all variables of the problem.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training. 
        If an int is given, the methode will use at least as many data points as the
        int. For grid sampling the number may be increased, to create a fitting grid.
        The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers. 
    sampling_strategy : str or dict
        The sampling strategy used to sample data points. Either a string that 
        sets the strategy for all variables or a dic giving a specific 
        strategy per variable. See domains for more details of useable strats.
    sampling_params : dict
        A dictionary containing additional parameters for the sampling.
        For each variable thats needs paramters the item of the dictionary needs
        to be a dictionary itself.
        E.g. if for the variable 'x' a normal distribution should be used. Then
        sampling_params = {'x': {'mean':...., 'cov':....}}.
    """
    def __init__(self, variables, dataset_size, sampling_strategy,
                 sampling_params={}):
        super().__init__(variables, dataset_size,
                         sampling_strategy, sampling_params)

    def get_data(self):
        data = {}
        # first find the wanted strats and divide the number of points 
        # accordingly 
        self.sampling_strategy = self._create_sampling_strat_dic()
        dataset_dic = self._create_dataset_dic()
        # create points over the methodes of the domains
        for vname in self.variables:
            # sample params can be None, therefore use .get(vname)
            sample_params = self.sampling_params.get(vname)
            data[vname] = self.variables[vname].domain.sample_inside(
                dataset_dic[vname],
                type=self.sampling_strategy[vname],
                sample_params=sample_params
            )
        # now order and maybe create needed meshgrids of the points
        data = self._reorder_datapoints(data)
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
        The sampling strategy used to sample points for the boundary variable.
        See domains for more details.
    sampling_params : dict
        A dictionary containing additional parameters for the sampling.
        For each variable thats needs paramters the item of the dictionary needs
        to be a dictionary itself.
        E.g. for the variable 'x' a normal distribution should be used. Then
        sampling_params = {'x': {'mean':...., 'cov':....}}.
    """    
    def __init__(self, variables, dataset_size, sampling_strategy, 
                 boundary_sampling_strategy, sampling_params={}):
        super().__init__(variables, dataset_size, sampling_strategy, 
                         sampling_params)
        self.boundary_sampling_strategy = boundary_sampling_strategy
        self.boundary_variable = None # set later 

    def get_data(self):
        data = {}
        # first find the wanted strats and divide the number of points 
        # accordingly 
        self.sampling_strategy = self._create_sampling_strat_dic()
        dataset_dic = self._create_dataset_dic()
        for vname in self.variables:
            sample_params = self.sampling_params.get(vname)
            if vname == self.boundary_variable:
                data[vname] = self.variables[vname].domain.sample_boundary(
                dataset_dic[vname],
                type=self.sampling_strategy[vname],
                sample_params=sample_params
            )
            else:
                data[vname] = self.variables[vname].domain.sample_inside(
                dataset_dic[vname],
                type=self.sampling_strategy[vname],
                sample_params=sample_params
            )
        data = self._reorder_datapoints(data)
        return data

    def _create_sampling_strat_dic(self):
        if isinstance(self.sampling_strategy, str):
            sampling_strat_dic = {}
            for vname in self.variables:
                if vname == self.boundary_variable:
                    sampling_strat_dic[vname] = self.boundary_sampling_strategy
                else:
                    sampling_strat_dic[vname] = self.sampling_strategy
            return sampling_strat_dic
        if isinstance(self.sampling_strategy, list):
            sampling_strat_dic = {}
            i = 0
            for vname in self.variables:
                if vname == self.boundary_variable:
                    sampling_strat_dic[vname] = self.boundary_sampling_strategy
                else:
                    sampling_strat_dic[vname] = self.sampling_strategy[i]
                    i += 1
            return sampling_strat_dic
        elif isinstance(self.sampling_strategy, dict):
            bound_strat = self.sampling_strategy.get(self.boundary_variable)
            if bound_strat is None:
                self.sampling_strategy[self.boundary_variable] = \
                    self.boundary_sampling_strategy
            return self.sampling_strategy
        else:
            raise TypeError(f"""Sampling strategy has to be a string, list or a
                            dictionary, got {type(self.sampling_strategy)}.""") 
    
    def _divide_dataset_for_int(self):
        """Splits the dataset size to all variables, depending
        on the dimension of the variable. 
        """
        if self._all_strats_random():
            return self._dataset_for_all_random(), True
        dataset_dic = {}
        sum_of_dim = self._compute_sum_of_dims()
        dataset_root = self._compute_dataset_size_root(sum_of_dim)
        for vname in self.variables:
            dim = self.variables[vname].domain.dim
            if vname == self.boundary_variable:
                data = self._set_data_for_boundary(dataset_root, dim)
                dataset_dic[vname] = data
            else:
                dataset_dic[vname] = dataset_root**dim
        return dataset_dic, False

    def _set_data_for_boundary(self, dataset_size, dim):
        # Only if the domain is a intervall we have to check 3
        # different cases
        check, value = self._check_boundary_strat()
        if dim == 1 & check:
            dataset_size = value
        else:
            dataset_size = dataset_size**dim
        return dataset_size

    def _check_boundary_strat(self):
        special_strats = ['lower_bound_only', 'upper_bound_only']
        if self.boundary_sampling_strategy == 'grid':
            return True, 2 # for grid (left and right bound of interval) we
                           # need to divide the whole dataset in two identical parts
        elif self.boundary_sampling_strategy in special_strats:
            return True, 1
        return False, None

    def _compute_dataset_size_root(self, sum_of_dim):
        boundary_dim = self.variables[self.boundary_variable].domain.dim
        check, value = self._check_boundary_strat()
        if boundary_dim == 1 & check:
            dataset_root = (self.dataset_size/value)**(1/(sum_of_dim-1))
        else:
            dataset_root = self.dataset_size**(1/sum_of_dim)
        return int(np.ceil(dataset_root))