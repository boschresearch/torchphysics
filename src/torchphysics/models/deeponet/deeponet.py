import torch

from torchphysics.problem.spaces.points import Points
from ..model import Model, Sequential
from .branchnets import BranchNet
from .trunknets import TrunkNet


class DeepONet(Model):
    """Implementation of the architecture used in the DeepONet paper [#]_.
    Consists of two single neural networks. One for the inputs of the function
    space (branch net) and one for the inputs of the variables (trunk net).

    Parameters
    ----------
    trunk_net : torchphysics.models.TrunkNet
        The neural network that will get the space/time/... variables as an
        input.
    branch_net : torchphysics.models.BranchNet
        The neural network that will get the function variables as an
        input.
    output_space : Space
        The space in which the final output of the DeepONet will belong to.
    output_neurons : int
        The number of output neurons, that will be the output of the
        TrunkNet and BranchNet. The corresponding outputs of both networks
        are then connected with the inner product.
        For higher dimensional outputs, will be multiplied my the dimension of
        the output space, so each dimension will have the same number of
        intermediate neurons.

    Notes
    -----
    The number of output neurons in the branch and trunk net have to be the same!

    ..  [#] Lu Lu and Pengzhan Jin and Guofei Pang and Zhongqiang Zhang
        and George Em Karniadakis, "Learning nonlinear operators via DeepONet
        based on the universal approximation theorem of operators", 2021
    """

    def __init__(self, trunk_net, branch_net, output_space, output_neurons):
        self._check_trunk_and_branch_correct(trunk_net, branch_net)
        super().__init__(input_space=trunk_net.input_space, output_space=output_space)
        self.trunk = trunk_net
        self.branch = branch_net
        self._finalize_trunk_and_branch(output_space, output_neurons)

    def _check_trunk_and_branch_correct(self, trunk_net, branch_net):
        """Checks if the trunk and branch net are compatible
        with each other.
        """
        if isinstance(trunk_net, Sequential):
            trunk_net = trunk_net.models[-1]
        assert isinstance(trunk_net, TrunkNet)
        assert isinstance(branch_net, BranchNet)

    def _finalize_trunk_and_branch(self, output_space, output_neurons):
        if isinstance(self.trunk, Sequential):
            self.trunk.models[-1].finalize(output_space, output_neurons)
        else:
            self.trunk.finalize(output_space, output_neurons)
        self.branch.finalize(output_space, output_neurons)

    def forward(self, trunk_inputs, branch_inputs=None, device="cpu"):
        """Apply the network to the given inputs.

        Parameters
        ----------
        trunk_inputs : torchphysics.spaces.Points
            The inputs for the trunk net.
        branch_inputs : callable, torchphysics.domains.FunctionSet, optional
            The function(s) for which the branch should be evaluaded. If no
            input is given, the branch net has to be fixed before hand!
        device : str, optional
            The device where the data lays. Default is 'cpu'.

        Returns
        -------
        torchphysics.spaces.Points
            A point object containing the output.

        """
        if not branch_inputs is None:
            self.fix_branch_input(branch_inputs, device=device)
        trunk_out = self.trunk(trunk_inputs)
        if len(trunk_out.shape) < 4:
            trunk_out = trunk_out.unsqueeze(0)  # shape = [1, trunk_n, dim, neurons]
        out = torch.sum(trunk_out * self.branch.current_out.unsqueeze(1), dim=-1)
        return Points(out, self.output_space)

    def _forward_branch(self, function_set, iteration_num=-1, device="cpu"):
        """Branch evaluation for training."""
        if iteration_num != function_set.current_iteration_num:
            function_set.current_iteration_num = iteration_num
            function_set.sample_params(device=device)
            discrete_fn_batch = self.branch._discretize_function_set(
                function_set, device=device
            )
            self.branch(discrete_fn_batch)

    def fix_branch_input(self, function, device="cpu"):
        """Fixes the branch net for a given function. this function will then be used
        in every following forward call. To set a new function just call this method
        again.

        Parameters
        ----------
        function : callable, torchphysics.domains.FunctionSet
            The function(s) for which the branch should be evaluaded.
        device : str, optional
            The device where the data lays. Default is 'cpu'.
        """
        self.branch.fix_input(function, device=device)
