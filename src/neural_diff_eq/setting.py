from typing import Iterable, OrderedDict
from pytorch_lightning import LightningDataModule as DataModule
import torch
import numpy as np

from .problem.problem import Problem
from .problem import Variable
from .problem.parameters.parameter import ParameterSub


class SimpleDataset(torch.utils.data.Dataset):
    """
    A dummy dataset which simply returns whole batches.
    Used to enable faster training with the whole dataset on a gpu.

    Parameters
    ----------
    data : A dictionary that contains batched data vor every condition
        and variables
    iterations : The number of iterations that should be performed in training.
    """
    def __init__(self, data, iterations):
        super().__init__()
        self.data = data
        self.epoch_len = iterations

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        return self.data


class Setting(Problem, DataModule):
    """
    A PDE setting, built of all variables and conditions in this problem.

    TODO: enable varying batches, i.e. non-full-batch-mode and maybe even a mode
    for datasets that are too large for gpu mem

    Parameters
    ----------
    variables : dict, tuple, list or Variable
        A collection of Variables for this DE Problem. The Domain of the Problem is
        the cartesian product of the domains of the variables.
    train_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are used in training
    val_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are tracked during validation
    parameters : list or dict of Parameter objects
        Parameters can be part of conditions and still be learned together with the
        solution during training.
    n_iterations : int
        Number of iterations per epoch.
    """
    def __init__(self, variables={}, train_conditions={}, val_conditions={},
                 parameters={}, n_iterations=1000):
        DataModule.__init__(self)
        self.n_iterations = n_iterations

        self.variables = OrderedDict()
        if isinstance(variables, (list, tuple)):
            for condition in variables:
                self.add_variable(condition)
        elif isinstance(variables, dict):
            for condition_name in variables:
                self.add_variable(variables[condition_name])
        elif isinstance(variables, Variable):
            self.add_variable(variables)
        else:
            raise TypeError(f"""Got type {type(variables)} but expected
                             one of list, tuple, dict or Variable.""")
        Problem.__init__(self,
                         train_conditions=train_conditions,
                         val_conditions=val_conditions)

        self.parameters = OrderedDict()
        if isinstance(parameters, (list, tuple)):
            for param in parameters:
                self.add_parameter(param)
        elif isinstance(parameters, dict):
            for param_name in parameters:
                self.add_parameter(parameters[param_name])
        elif isinstance(parameters, ParameterSub):
            self.add_parameter(parameters)
        else:
            raise TypeError(f"""Got type {type(parameters)} but expected
                             one of list, tuple, dict or ParameterSub.""")
        # run data preparation manually
        self.prepare_data()

    """Methods to load data with lightning"""

    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        self.train_data = {}
        train_conditions = self.get_train_conditions()
        for name in train_conditions:
            self.train_data[name] = train_conditions[name].get_data()

        self.val_data = {}
        val_conditions = self.get_val_conditions()
        for name in val_conditions:
            self.val_data[name] = val_conditions[name].get_data()

    def setup(self, stage=None):
        print('setup: ', self.trainer.model.device)
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        if stage == 'fit' or stage is None:
            self.train_data = self._setup_stage(
                self.get_train_conditions(), self.train_data)
        if stage == 'validate' or stage is None:
            self.val_data = self._setup_stage(
                self.get_val_conditions(), self.val_data)

    def _setup_stage(self, conditions, data):
        for cn in data:
            if isinstance(data[cn], dict):
                # only input data is given
                data[cn] = self._setup_input_data(data[cn],
                                                  conditions[cn].track_gradients)
            elif len(data[cn]) == 2:
                # pairs of inputs and targets are given
                data_dic, target = data[cn]
                data_dic = self._setup_input_data(data_dic,
                                                  conditions[cn].track_gradients)
                target = self._setup_target_data(target)
                data[cn] = data_dic, target
            else:  # triple of inputs, targets and normals are given
                data_dic, target, normals = data[cn]
                data_dic = self._setup_input_data(data_dic,
                                                  conditions[cn].track_gradients)
                target = self._setup_target_data(target)
                normals = self._setup_target_data(normals)
                data[cn] = data_dic, target, normals
        return data

    def _setup_target_data(self, target):
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        return target

    def _setup_input_data(self, data, track_gradients):
        for vn in data:
            if isinstance(data[vn], np.ndarray):
                data[vn] = torch.from_numpy(data[vn])

            # enable gradient tracking if necessary
            if isinstance(track_gradients, bool):
                data[vn].requires_grad = track_gradients
            elif isinstance(track_gradients, Iterable):
                data[vn].requires_grad = vn in track_gradients
            else:
                raise TypeError(
                    f'track_gradients of {vn} should be either bool or iterable.')
        return data

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.train_data, self.n_iterations),
            batch_size=None,
            num_workers=0
        )
        return dataloader

    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.val_data, 1),
            batch_size=None,
            num_workers=0
        )
        return dataloader

    """Problem methods"""

    def add_variable(self, variable):
        """Adds a new Variable object to the Setting, registers the variable and its conditions
        """
        assert isinstance(variable, Variable), f"{variable} should be a Variable obj."
        self.variables[variable.name] = variable
        # register the variable in this setting
        variable.setting = self
        # update its condition variables
        for cname in variable.train_conditions:
            variable.train_conditions[cname].setting = self
        for cname in variable.val_conditions:
            variable.val_conditions[cname].setting = self

    def add_parameter(self, param):
        assert isinstance(param, ParameterSub), f"{param} should be a ParameterSub obj."
        self.parameters[param.get_name()] = param

    def add_train_condition(self, condition, boundary_var=None):
        """Adds and registers a condition that is used during training
        """
        if boundary_var is None:
            assert condition.name not in self.train_conditions, \
                f"{condition.name} cannot be present twice."
            condition.setting = self
            self.train_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_train_condition(condition)

    def add_val_condition(self, condition, boundary_var=None):
        """Adds and registers a condition that is used for validation
        """
        if boundary_var is None:
            assert condition.name not in self.val_conditions, \
                f"{condition.name} cannot be present twice."
            condition.setting = self
            self.val_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_val_condition(condition)

    def get_train_conditions(self):
        """Returns all training conditions present in this problem.
        This does also include conditions in the variables.
        """
        dct = {}
        for cname in self.train_conditions:
            dct[cname] = self.train_conditions[cname]
        for vname in self.variables:
            vconditions = self.variables[vname].get_train_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def get_val_conditions(self):
        """Returns all validation conditions present in this problem.
        This does also include conditions in the variables.
        """
        dct = {}
        for cname in self.val_conditions:
            dct[cname] = self.val_conditions[cname]
        for vname in self.variables:
            vconditions = self.variables[vname].get_val_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def is_well_posed(self):
        raise NotImplementedError

    def serialize(self):
        dct = {}
        dct['name'] = 'Setting'
        dct['n_iterations'] = self.n_iterations
        v_dict = {}
        for v_name in self.variables:
            v_dict[v_name] = self.variables[v_name].serialize()
        dct['variables'] = v_dict
        t_c_dict = {}
        for c_name in self.train_conditions:
            t_c_dict[c_name] = self.train_conditions[c_name].serialize()
        dct['train_conditions'] = t_c_dict
        v_c_dict = {}
        for c_name in self.val_conditions:
            v_c_dict[c_name] = self.val_conditions[c_name].serialize()
        dct['val_conditions'] = v_c_dict
        return dct
