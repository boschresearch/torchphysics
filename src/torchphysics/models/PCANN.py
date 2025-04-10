import torch

from .model import Model
from .fcn import _construct_FC_layers
from ..problem.domains.functionsets import DiscreteFunctionSet
from ..problem.spaces.points import Points

class PCANN(Model):
    """A general neural network model that uses PCA to reduce the 
    dimensionality of the input and output spaces and then only learns 
    a mapping of the principal components. 
    Follows the idea presented in [1].

    Parameters
    ----------
    input_space, output_space : Space
        The input and output space of the model.
    pca_in, pca_out : tuple
        The principal component decomposition of the input and output data. 
        Should be of the form (U, S, V) where U contains the left eigen vectors,
        S the eigen values of the covariance matrix and V the right eigen 
        vectors of input and output data respectively.
        See also https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html  
        for the format which is expected.
    output_shape : tuple
        The shape of the output data. 
        This is needed to reshape the output after the neural network and PCA
        have been applied.
    mean_in, mean_out : tensor
        The mean of the input and output data. 
        This is used to normalize the data before applying the PCA.
    std_in, std_out : tensor
        The standard deviation of the input and output data. 
        This is used to normalize the data before applying the PCA.

    Notes 
    -----
    The default implemwentation flatten the input and output data along all 
    dimensions except the first one.

    ..  [1] Kaushik Bhattacharya et al., "Model Reduction And Neural Networks For 
            Parametric PDEs", 2021
    """

    def __init__(self, input_space, output_space, pca_in, pca_out, output_shape,
                 mean_in=torch.tensor(0), mean_out=torch.tensor(0), 
                 std_in=torch.tensor(1), std_out=torch.tensor(1)):
        super().__init__(input_space, output_space)
        
        self.register_buffer("eigenvectors_in", pca_in[2])
        self.register_buffer("eigenvectors_out", pca_out[2])

        ev_values_in = torch.sqrt(pca_in[1]**2 / (len(pca_in[0]) - 1))
        self.register_buffer("eigenvalues_in", ev_values_in)
        ev_values_out = torch.sqrt(pca_out[1]**2 / (len(pca_out[0]) - 1))
        self.register_buffer("eigenvalues_out", ev_values_out)

        self.register_buffer("mean_in", mean_in)
        self.register_buffer("mean_out", mean_out)
        self.register_buffer("std_in", std_in)
        self.register_buffer("std_out", std_out)

        self.output_shape = [-1, *output_shape, self.output_space.dim]


    @classmethod
    def from_fn_set(cls, 
                    input_fn_set : DiscreteFunctionSet, 
                    output_fn_set : DiscreteFunctionSet):
        """Construct a PCANN model from two **discrete** function sets.

        Parameters
        ----------
        input_fn_set, output_fn_set : DiscreteFunctionSet
            The function sets containing the input and expected output
            data. 
        """
        return cls(input_fn_set.function_space.output_space, 
                   output_fn_set.function_space.output_space, 
                   output_shape = output_fn_set.data_shape,
                   pca_in = input_fn_set.principal_components, 
                   pca_out = output_fn_set.principal_components, 
                   mean_in = input_fn_set.mean, 
                   mean_out = output_fn_set.mean, 
                   std_in = input_fn_set.std,
                   std_out = output_fn_set.std)


    def apply_network(self, pc_input):
        """Apply the neural network to the principal components.
        
        Parameters
        ----------
        pc_input : tensor
            The principal components of the input data.

        Returns
        -------
        tensor
            The predicted principal components of the output data.

        Note 
        ----
        This function should be implemented in the sub classes.
        """
        raise NotImplementedError("PCANN can not be used directly! Use one of the sub classes.")


    def forward(self, points):
        if not torch.is_tensor(points):
            points = self._fix_points_order(points).as_tensor
        # normalize inputs 
        points = (points - self.mean_in) / self.std_in
        # apply pca
        points = torch.flatten(points, start_dim=1)
        pc_in = points @ self.eigenvectors_in
        pc_in /= self.eigenvalues_in
        # Then evaluate neural network
        pc_out = self.apply_network(pc_input=pc_in)
        # "inverse" pca
        pc_out *= self.eigenvalues_out
        points_out = pc_out @ self.eigenvectors_out.T
        points_out = points_out.reshape(self.output_shape)
        # "inverse" normalization
        points_out = points_out * self.std_out + self.mean_out
        # reshape and transform to points
        return Points(points_out, self.output_space)


class PCANN_FC(PCANN):
    """ A PCANN model that uses a fully connected neural network to learn the
    mapping between the principal components of the input and output data.
    
    Parameters
    ----------
    Parameters
    ----------
    input_space, output_space : Space
        The input and output space of the model.
    pca_in, pca_out : tuple
        The principal component decomposition of the input and output data. 
        Should be of the form (U, S, V) where U contains the left eigen vectors,
        S the eigen values of the covariance matrix and V the right eigen 
        vectors of input and output data respectively.
        See also https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html  
        for the format which is expected.
    output_shape : tuple
        The shape of the output data. 
        This is needed to reshape the output after the neural network and PCA
        have been applied.
    hidden : list or tuple
        The number and size of the hidden layers of the neural network.
        The lenght of the list/tuple will be equal to the number
        of hidden layers, while the i-th entry will determine the number
        of neurons of each layer.
        E.g hidden = (10, 5) -> 2 layers, with 10 and 5 neurons.
    activations : torch.nn or list, optional
        The activation functions of this network. If a single function is passed
        as an input, will use this function for each layer.
        If a list is used, will use the i-th entry for i-th layer.
        Deafult is nn.Tanh().
    xavier_gains : float or list, optional
        For the weight initialization a Xavier/Glorot algorithm will be used.
        The gain can be specified over this value.
        Default is 5/3.
    mean_in, mean_out : tensor
        The mean of the input and output data. 
        This is used to normalize the data before applying the PCA.
    std_in, std_out : tensor
        The standard deviation of the input and output data. 
        This is used to normalize the data before applying the PCA.
    """
    def __init__(self, input_space, output_space, pca_in, pca_out, output_shape,
                 hidden=(20, 20, 20), activations=torch.nn.Tanh(),xavier_gains=5/3,
                 mean_in=torch.tensor(0), mean_out=torch.tensor(0), 
                 std_in=torch.tensor(1), std_out=torch.tensor(1)):
        super().__init__(input_space, output_space, pca_in, pca_out, output_shape,
                         mean_in, mean_out, std_in, std_out)
        
        layers = _construct_FC_layers(
            hidden=hidden,
            input_dim=len(self.eigenvalues_in),
            output_dim=len(self.eigenvalues_out),
            activations=activations,
            xavier_gains=xavier_gains,
        )
    
        self.sequential = torch.nn.Sequential(*layers)

    @classmethod
    def from_fn_set(cls, 
                    input_fn_set : DiscreteFunctionSet, 
                    output_fn_set : DiscreteFunctionSet, 
                    hidden=(20, 20, 20), 
                    activations=torch.nn.Tanh(),
                    xavier_gains=5/3):
        """Construct a PCANN_FC model from two **discrete** function sets.

        Parameters
        ----------
        input_fn_set, output_fn_set : DiscreteFunctionSet
            The function sets containing the input and expected output
            data. 
        hidden : list or tuple
            The number and size of the hidden layers of the neural network.
            The lenght of the list/tuple will be equal to the number
            of hidden layers, while the i-th entry will determine the number
            of neurons of each layer.
            E.g. hidden = (10, 5) -> 2 layers, with 10 and 5 neurons.
        activations : torch.nn or list, optional
            The activation functions of this network. If a single function is passed
            as an input, will use this function for each layer.
            If a list is used, will use the i-th entry for i-th layer.
            Deafult is nn.Tanh().
        xavier_gains : float or list, optional
            For the weight initialization a Xavier/Glorot algorithm will be used.
            The gain can be specified over this value.
            Default is 5/3.
        """
        return cls(input_fn_set.function_space.output_space, 
                   output_fn_set.function_space.output_space, 
                   output_shape = output_fn_set.data_shape,
                   pca_in = input_fn_set.principal_components, 
                   pca_out = output_fn_set.principal_components, 
                   hidden=hidden, activations=activations, 
                   xavier_gains=xavier_gains,
                   mean_in = input_fn_set.mean, 
                   mean_out = output_fn_set.mean, 
                   std_in = input_fn_set.std,
                   std_out = output_fn_set.std)
    

    def apply_network(self, pc_input):
        return self.sequential(pc_input)