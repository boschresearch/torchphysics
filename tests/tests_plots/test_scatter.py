from torchphysics.problem.spaces import R1, R2
from torchphysics.problem.domains import Interval, Circle
from torchphysics.problem.samplers import RandomUniformSampler, GridSampler
from torchphysics.utils import scatter


def test_scatter_1D():
    I = Interval(R1('t'), 0, 1)
    ips = GridSampler(I, n_points=20)
    bps = GridSampler(I.boundary_left, n_points=1)
    _ = scatter(R1('t'), ips, bps)


def test_scatter_2D():
    C = Circle(R2('x'), [0, 0], 1)
    ips = RandomUniformSampler(C, n_points=250)
    bps = RandomUniformSampler(C.boundary, n_points=100)
    _ = scatter(R2('x'), ips, bps)


def test_scatter_3D():
    C = Circle(R2('x'), [0, 0], 1)
    I = Interval(R1('t'), 0, 1)
    ips = GridSampler(C, n_points=20) * GridSampler(I, n_points=5)
    bps = GridSampler(C.boundary, n_points=10) * GridSampler(I.boundary_left, n_points=1)
    _ = scatter(R2('x')*R1('t'), ips, bps)