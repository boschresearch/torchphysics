import torch
import math

from ...spaces.points import Points
from .functionset import TestFunctionSet
from ..domain1D import Interval


class HarmonicFunctionSet1D(TestFunctionSet):

    def __init__(self, function_space, interval : Interval, frequence, 
                 samples_per_max_frequence : int = 5):
        super().__init__(function_space=function_space)
        self.interval = interval
        self.samples_max = samples_per_max_frequence

        if isinstance(frequence, list):
            self.basis_dim = max(frequence)
            self.frequence_list = frequence
        else:
            self.basis_dim = frequence
            self.frequence_list = torch.arange(1, frequence+1, 1)

        quad_points = torch.linspace(self.interval.lower_bound(), self.interval.upper_bound(), 
                                     self.basis_dim * self.samples_max + 2)[1:-1]
        self.quadrature_points_per_dof = quad_points.repeat((self.basis_dim, 1)).unsqueeze(-1)
        self.quadrature_weights_per_dof = quad_points[1] - quad_points[0]

        self.compute_basis_at_quadrature_points()


    def switch_quadrature_mode_on(self, set_on : bool):
        self.quadrature_mode_on = set_on
        if not set_on:
            AssertionError("Arbritrary evaluation not implemented!")


    def to(self, device):
        self.quadrature_points_per_dof = self.quadrature_points_per_dof.to(device)
        self.quadrature_weigths_per_dof = self.quadrature_weights_per_dof.to(device)
        self.basis_at_quadrature = self.basis_at_quadrature.to(device)
        self.grad_at_quadrature = self.grad_at_quadrature.to(device)


    def compute_basis_at_quadrature_points(self):
        self.basis_at_quadrature = torch.zeros_like(self.quadrature_points_per_dof)
        self.grad_at_quadrature = torch.zeros_like(self.quadrature_points_per_dof)

        int_size = self.interval.upper_bound() - self.interval.lower_bound()
        for i, n in enumerate(self.frequence_list):
            self.basis_at_quadrature[i] = \
                torch.sin(n*math.pi/(int_size) * \
                          (self.quadrature_points_per_dof[i] - self.interval.lower_bound()))
            self.grad_at_quadrature[i] = -n*math.pi/(int_size) * \
                torch.cos(n*math.pi/(int_size) * \
                          (self.quadrature_points_per_dof[i] - self.interval.lower_bound()))
    

    def __call__(self, x=None):
        if self.quadrature_mode_on:
            input_variable_name = self.function_space.input_space.variables.pop()
            return Points(self.eval_fn_helper.apply(x[input_variable_name], 
                                                    self.basis_at_quadrature, 
                                                    self.grad_at_quadrature), 
                          self.function_space.output_space)
        else:
            raise NotImplementedError


    def grad(self, x=None):
        if self.quadrature_mode_on or x == None:
            return self.grad_at_quadrature


    def get_quad_weights(self, n):
        repeats = n // len(self.quadrature_weights_per_dof)
        return self.quadrature_weights_per_dof.repeat((repeats, 1, 1))


    def get_quadrature_points(self):
        return Points(self.quadrature_points_per_dof, self.function_space.input_space)