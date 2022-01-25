"""Contains a class which extracts the needed arguments of an arbitrary 
methode/function and wraps them for future usage. E.g correctly choosing 
the needed arguments and passing them on to the original function.
"""
import inspect
import copy 
import torch

from ..problem.spaces.points import Points


class UserFunction:
    """Wraps a function, so that it can be called with arbitrary input arguments.
    
    Parameters
    ----------
    fun : callable
        The original function that should be wrapped.
    defaults : dict, optional
        Possible defaults arguments of the function. If none are specified will
        check by itself if there are any. 
    args : dict, optional
        All arguments of the function. If none are specified will
        check by itself if there are any. 

    Notes
    -----
    Uses inspect.getfullargspec(fun) to get the possible input arguments.
    When called just extracts the needed arguments and passes them to the 
    original function. 
    """
    def __init__(self, fun, defaults={}, args={}):
        if isinstance(fun, (UserFunction, DomainUserFunction)):
            self.fun = fun.fun
            self.defaults = fun.defaults
            self.args = fun.args
        else:
            self._transform_to_user_function(fun, defaults, args)

    def _transform_to_user_function(self, fun, defaults, args):
        self.fun = fun
        self.defaults = defaults
        self.args = args
        if callable(self.fun) and self.defaults == {} and self.args == {}:
            self._set_input_args_for_function()

    def _set_input_args_for_function(self):
        f_args = inspect.getfullargspec(self.fun).args

        # we check that the function defines all needed parameters
        if inspect.getfullargspec(self.fun).varargs is not None or \
            inspect.getfullargspec(self.fun).varkw is not None:
            raise ValueError("""
                             Variable arguments are not supported in
                             UserFunctions. Please use keyword arguments.
                             """)

        f_defaults = inspect.getfullargspec(self.fun).defaults
        f_kwonlyargs = inspect.getfullargspec(self.fun).kwonlyargs
        #f_kwonlydefaults = inspect.getfullargspec(self.fun).kwonlydefaults
        # NOTE: By above check, there should not be kwonlyargs. However, we still catch
        # this case here.
        self.args = f_args + f_kwonlyargs

        # defaults always align at the end of the args
        self.defaults = {}
        if not f_defaults is None:
            self.defaults = {self.args[-i]: f_defaults[-i] 
                             for i in range(len(f_defaults), 0, -1)}
        #if not f_kwonlydefaults is None:
        #    self.defaults.update(f_kwonlydefaults)

    def __call__(self, args={}, vectorize=False):
        """To evalute the function. Will automatically extract the needed arguments 
        from the input data and will set the possible default values.

        Parameters
        ----------
        args : dict or torchphysics.Points
            The input data, where the function should be evaluated.
        vectorize : bool, optional
            If the original function can work with a batch of data, or
            a loop needs to be used to evaluate the function.
            default is False, which means that we assume the function
            can work with a batch of data.

        Returns
        -------
        torch.tensor
            The output values of the function.
        """
        if isinstance(args, Points):
            args = args.coordinates
        # check that every necessary arg is given
        for key in self.necessary_args:
            assert key in args, \
                f"The argument '{key}' is necessary in {self.__name__} but not given."
        # if necessary, pass defaults
        inp = {key: args[key] for key in self.args if key in args}
        inp.update({key: self.defaults[key] for key in self.args if key not in args})
        if not vectorize:
            return self.evaluate_function(**inp)
        else:
            return self.apply_to_batch(inp)

    def evaluate_function(self, **inp):
        """Evaluates the original input function. Should not be used directly, 
        rather use the call-methode.
        """
        if callable(self.fun):
            return self.fun(**inp)
        return self.fun

    def apply_to_batch(self, inp):
        """Apply the function to a batch of elements by running a for-loop.
        we assume that all inputs either have batch (i.e. maximum) dimension or
        are a constant param.

        Parameters
        ----------
        inp : torchphysics.points
            The Points-object of the input data

        Returns
        -------
        torch.tensor
            The output values of the function, for each input.

        """
        batch_size = max(len(inp[key]) for key in inp)
        out = []
        for i in range(batch_size):
            inp_i = {}
            for key in inp:
                if len(inp[key]) == batch_size:
                    inp_i[key] = inp[key][i]
                else:
                    inp_i[key] = inp[key]
            o = self.fun(**inp_i)
            if o is not None:
                out.append(o)
        return out

    def partially_evaluate(self, **args):
        """(partially) evaluates a given function.

        Parameters
        ----------
        **args :
            The arguments where the function should be (partially) evaluated.

        Returns
        -------
        Out : value or UserFunction
            If the input arguments are enough to evalate the whole function, the 
            corresponding output is returned. 
            If some needed arguments are missing, a copy of this UserFunction will 
            be returned. Whereby the values of **args will be added to the 
            default values of the returned UserFunction.
        """
        if callable(self.fun):
            if all(arg in args for arg in self.necessary_args):
                inp = {key: args[key] for key in self.args if key in args}
                inp.update({key: self.defaults[key] for key in self.args if key not in args})
                return self.fun(**inp)
            else:
                # to avoid manipulation of given param obj, we create a copy
                copy_self = copy.deepcopy(self)
                copy_self.set_default(**args)
                return copy_self
        return self.fun

    def __name__(self):
        """The name of the function

        Returns
        -------
        str
            The name of the function
        """
        return self.fun.__name__

    def set_default(self, **args):
        """Sets a input argument to given value.

        Parameters
        ----------
        **args:
            The value the input should be set to.
        """
        self.defaults.update({key: args[key] for key in args if key in self.args})

    def remove_default(self, *args, **kwargs):
        """Removes an default value of a input argument.

        Parameters
        ----------
        *args, **kwargs:
            The arguments for which the default values should be deleted.
        """
        for key in args:
            self.defaults.pop(key)
        for key in kwargs.keys():
            self.defaults.pop(key)

    def __deepcopy__(self, memo):
        """Creates a copy of the function
        """
        cls = self.__class__
        copy_object = cls.__new__(cls, self.fun)
        memo[id(self)] = copy_object
        for k, v in self.__dict__.items():
            setattr(copy_object, k, copy.deepcopy(v, memo))
        return copy_object

    @property
    def necessary_args(self):
        """Returns the function arguments that are needed to evaluate this function.

        Returns
        -------
        list :
            The needed arguments.
        """
        return [arg for arg in self.args if arg not in self.defaults]

    @property
    def optional_args(self):
        """Returns the function arguments that are optional to evaluate this function.

        Returns
        -------
        list :
            The optional arguments.
        """
        return [arg for arg in self.args if arg in self.defaults]


class DomainUserFunction(UserFunction):
    """Extension of the original UserFunctions, that are used in the Domain-Class.
    
    Parameters
    ----------
    fun : callable
        The original function that should be wrapped.
    defaults : dict, optional
        Possible defaults arguments of the function. If none are specified will
        check by itself if there are any. 
    args : dict, optional
        All arguments of the function. If none are specified will
        check by itself if there are any. 

    Notes
    -----
    The only difference to normal UserFunction is how the evaluation 
    of the original function is handled. Since all Domains use Pytorch, 
    we check that the output always is a torch.tensor. In the case that the function
    is not constant, we also append an extra dimension to the output, so that the 
    domains can work with it correctly. 
    """
    def __call__(self, args={}, device='cpu'):
        """To evalute the function. Will automatically extract the needed arguments 
        from the input data and will set the possible default values.

        Parameters
        ----------
        args : dict or torchphysics.Points
            The input data, where the function should be evaluated.
        device : str, optional
            The device on which the output of th efunction values should lay.
            Default is 'cpu'.

        Returns
        -------
        torch.tensor
            The output values of the function.
        """
        if isinstance(args, Points):
            args = args.coordinates
        if len(args) != 0: # set the device correctly
            device = args[list(args.keys())[0]].device
        # check that every necessary arg is given
        for key in self.necessary_args:
            assert key in args, \
                f"The argument '{key}' is necessary in {self.__name__} but not given."
        # if necessary, pass defaults
        inp = {key: args[key] for key in self.args if key in args}
        inp.update({key: self.defaults[key] for key in self.args if key not in args})
        return self.evaluate_function(device=device, **inp)

    def evaluate_function(self, device='cpu', **inp):
        """Evaluates the original input function. Should not be used directly, 
        rather use the call-methode.

        Parameters
        ----------
        device : str, optional
            The device on which the output of th efunction values should lay.
            Default is 'cpu'.
        inp 
            The input values.
        """
        if callable(self.fun):
            fun_eval = self.fun(**inp)
            if not isinstance(fun_eval, torch.Tensor):
                fun_eval = torch.tensor(fun_eval, device=device)
            return fun_eval[:, None]
        else:
            if isinstance(self.fun, torch.Tensor):
                self.fun = self.fun.to(device)
                return self.fun
            else: 
                return torch.tensor(self.fun, device=device).float()