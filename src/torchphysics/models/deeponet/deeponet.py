import torch

from torchphysics.problem.spaces.points import Points
from ..model import Model, Sequential
from .subnets import BranchNet, TrunkNet


class DeepONet(Model):

    def __init__(self, trunk_net, branch_net, function_set):
        self._check_trunk_and_branch_correct(trunk_net, branch_net)
        super().__init__(input_space=trunk_net.input_space, 
                         output_space=trunk_net.output_space)
        self.trunk = trunk_net
        self.branch = branch_net
        self.function_set = function_set
        self.current_iteration_num = -1

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
        return Points(torch.mm(self.branch.current_out, trunk_out.T).reshape(-1, 1), 
                      self.output_space)

    def _forward_branch(self, device='cpu', iteration_num=-1):
        """Branch evaluation for training.
        """
        if iteration_num > self.current_iteration_num:
            self.current_iteration_num = iteration_num
            self.function_set.sample_params(device=device)
            discrete_fn_batch = self.branch._discretize_function_set(self.function_set, 
                                                                     device=device)
            self.branch(discrete_fn_batch)
        # what if new trainig gets strated (LBFGS after Adam)? At start 
        # we need to reset self.current_iteration_num

    def fix_branch_input(self, function, device='cpu'):
        self.branch.fix_input(function, device=device)