"""Function to show an example of the created points of the sampler. 
"""
import numpy as np

import matplotlib.pyplot as plt


def scatter(subspace, *samplers):
    """Shows (one batch) of used points in the training. If the sampler is
    static, the shown points will be the points for the training. If not
    the points may vary, depending of the sampler. 

    Parameters
    ----------
    subspace : torchphysics.problem.Space
        The (sub-)space of which the points should be plotted.
        Only plotting for dimensions <= 3 is possible.
    *samplers : torchphysics.problem.Samplers
        The diffrent samplers for which the points should be plotted.
        The plot for each sampler will be created in the order there were
        passed in.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        The figure handle of the plot.
    """
    assert subspace.dim <= 3, "Can only scatter points in dimensions <= 3."

    fig, ax, scatter_fn = _choose_scatter_function(subspace.dim)
    ax.grid()
    for sampler in samplers:
        points = sampler.sample_points()[:, list(subspace.keys())]
        numpy_points = points.as_tensor.detach().cpu().numpy()
        labels = _create_labels(subspace)
        scatter_fn(ax, numpy_points, labels)
    return fig

def _create_labels(subspace):
    labels = []
    for var in subspace:
        if subspace[var] == 1:
            labels.append(var)
        else:
            for i in range(subspace[var]):
                labels.append(var+f'_{i+1}')
    return labels

def _choose_scatter_function(space_dim):
    fig = plt.figure()
    if space_dim == 1:
        ax = fig.add_subplot()
        return fig, ax, _scatter_1D
    elif space_dim == 2:
        ax = fig.add_subplot()
        return fig, ax, _scatter_2D    
    else:
        ax = fig.add_subplot(projection='3d')
        return fig, ax, _scatter_3D  


def _scatter_1D(ax, points, labels):
    ax.scatter(points, np.zeros_like(points))
    ax.set_xlabel(labels[0])


def _scatter_2D(ax, points, labels):
    ax.scatter(points[:, 0], points[:, 1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


def _scatter_3D(ax, points, labels):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])