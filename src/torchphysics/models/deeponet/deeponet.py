import torch

from torchphysics.problem.spaces.points import Points
from ..model import Model, Sequential
from .subnets import BranchNet, TrunkNet


class DeepONet(Model):

    def __init__(self, trunk_net, branch_net):
        self._check_trunk_and_branch_correct(trunk_net, branch_net)
        super().__init__(input_space=trunk_net.input_space, 
                         output_space=trunk_net.output_space)
        self.trunk = trunk_net
        self.branch = branch_net

    def _check_trunk_and_branch_correct(self, trunk_net, branch_net):
        """Just checks if the trunk and branch net are compatible
        with each other.
        """
        if isinstance(trunk_net, Sequential):
            trunk_net = trunk_net.models[-1]
        assert isinstance(trunk_net, TrunkNet)
        assert isinstance(branch_net, BranchNet)
        assert trunk_net.output_space == branch_net.output_space
        assert trunk_net.output_neurons == branch_net.output_neurons

    def forward(self, trunk_inputs, branch_inputs=None, device='cpu'):
        if branch_inputs:
            self.fix_branch_input(branch_inputs, device=device)
        trunk_out = self.trunk(trunk_inputs)
        return Points(torch.matmul(trunk_out, self.branch.current_out.unsqueeze(-1)).unsqueeze(-1),
                      self.output_space)

    def _forward_branch(self, function_set, iteration_num=-1, device='cpu'):
        """Branch evaluation for training.
        Assume branch input points have already been sampled.
        """
        if iteration_num != function_set.current_iteration_num:
            function_set.current_iteration_num = iteration_num
            function_set.sample_params(device=device)
            discrete_fn_batch = self.branch._discretize_function_set(function_set, device=device)
            self.branch(discrete_fn_batch)

    def fix_branch_input(self, function, device='cpu'):
        self.branch.fix_input(function, device=device)