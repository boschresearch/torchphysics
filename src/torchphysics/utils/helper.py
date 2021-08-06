"""
Collection of useful helper functions.
"""
from typing import Iterable
import inspect
from inspect import signature

import numpy as np

# TODO: These methods should be reimplemented in a clearer way.
# It would be better to seperate all performed steps into single
# methods that are called by summarizing methods.
# E.g. a function that only prepares the inputs for some other function.
# And then a function that prepares everything for a data function.
# This is too confusing by now.


def is_batch(arg, batch_size):
    # NOTE: this is not 100% safe, since in theory, some arguments
    #       could consist of batch_size elements
    #       but for now, we'll keep the length as criterion
    return isinstance(arg, np.ndarray) and len(arg) == batch_size


def apply_to_batch(f, batch_size, **batch):
    """
    Applies a given function that is defined point-wise to a dictionary,
    that also contains numpy.ndarray-batches of points.
    This is not defined for torch tensors (yet). It is quite inefficient,
    so it should exclusively be used for computations that are performed
    only once during training, e.g. data setup.
    """
    assert batch_size is not None, "batch_size should be specified!"

    # prepare output array
    pass_dict = {}
    for v in batch:
        if is_batch(batch[v], batch_size):
            pass_dict[v] = batch[v][0]
        else:
            pass_dict[v] = batch[v]
    out_0 = f(**pass_dict)
    if isinstance(out_0, Iterable):
        out_len = len(out_0)
    else:
        out_len = 1

    out = np.zeros((batch_size, out_len))

    # evaluate given function
    for i in range(batch_size):
        pass_dict = {}
        for v in batch:
            if is_batch(batch[v], batch_size):
                pass_dict[v] = batch[v][i]
            else:
                pass_dict[v] = batch[v]
        o = f(**pass_dict)
        if o is None:
            out[i, :] = np.nan
        else:
            out[i, :] = o
    return out.astype(np.float32)


def prepare_user_fun_input(fun, args):
    # first case: the user function can take arbitrary args:
    if '**' in str(signature(fun)):
        return args
    # second case: we only pass the args needed by the user function
    else:
        inp = {}
        for k in dict(signature(fun).parameters):
            try:
                inp[k] = args[k]
            except KeyError:
                if dict(signature(fun).parameters)[k].default == inspect._empty:
                    print(f"""The user-defined function '{fun.__name__}' expects arguments
                              {str(signature(fun))}.
                              However, only {args.keys()} are given in the library.""")
                    raise KeyError
        return inp
