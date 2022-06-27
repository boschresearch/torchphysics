"""Different plotting functions:

- ``plot_functions`` implement functions to show the output of the neural network
  or values derivative from it (derivatives, ...).
- ``animation`` implement the same concepts as the plot functions, just as animations.
- ``scatter_points`` are meant to show a batch of used training points of a sampler. 

"""

from .plot_functions import plot
from .animation import animate
from .scatter_points import scatter