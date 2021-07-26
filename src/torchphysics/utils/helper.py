"""
Collection of useful helper functions.
"""
from typing import Iterable
from inspect import signature

import numpy as np


def apply_to_batch(f, batch_size, **batch):
    """
    Applies a given function that is defined point-wise to a dictionary,
    that also contains numpy.ndarray-batches of points.
    This is not defined for torch tensors (yet). It is quite inefficient,
    so it should exclusively be used for computations that are performed
    only once during training, e.g. data setup.
    """
    assert batch_size is not None, "batch_size should be specified!"

    def is_batch(arg):
        return isinstance(arg, np.ndarray) and len(batch[v]) == batch_size

    # prepare output array
    pass_dict = {}
    for v in batch:
        if is_batch(batch[v]):
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
            if is_batch(batch[v]):
                # NOTE: this is not 100% safe, since in theory, some arguments
                #       could consist of batch_size elements
                #       but for now, we'll keep the length as criterion
                pass_dict[v] = batch[v][i]
            else:
                pass_dict[v] = batch[v]
        out[i, :] = f(**pass_dict)
    return out.astype(np.float32)


def apply_user_fun(f, args, whole_batch=True, batch_size=None):
    """
    helper method that passes only the required arguments to
    a user-defined function
    """
    # first case: the user function can take arbitrary args:
    if '**' in str(signature(f)):
        if whole_batch:
            return f(**args)
        else:
            return apply_to_batch(f, batch_size=batch_size, **args)
    # second case: we only pass the args needed by the user function
    else:
        try:
            inp = {k: args[k] for k in dict(signature(f).parameters)}
            if whole_batch:
                return f(**inp)
            else:
                return apply_to_batch(f, batch_size=batch_size, **inp)
        except KeyError:
            print(f"""The user-defined function '{f.__name__}' expects arguments {str(signature(f))}.
                      However, only {args.keys()} are given in the library.""")
        raise KeyError
