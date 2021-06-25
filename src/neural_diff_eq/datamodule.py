from typing import Iterable
import pytorch_lightning as pl
import torch


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


class ProblemDataModule(pl.LightningDataModule):
    """
    TODO: enable varying batches, i.e. non-full-batch-mode and maybe even a mode
    for datasets that are too large for gpu mem
    """
    def __init__(self, problem, n_iterations):
        super().__init__()
        self.problem = problem
        self.n_iterations = n_iterations

    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        self.train_data = {}
        train_conditions = self.problem.get_train_conditions()
        for name in train_conditions:
            self.train_data[name] = train_conditions[name].get_data()

        self.val_data = {}
        val_conditions = self.problem.get_val_conditions()
        for name in val_conditions:
            self.val_data[name] = val_conditions[name].get_data()

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        if stage == 'fit' or stage is None:
            self.train_data = self._setup_stage(
                self.problem.get_train_conditions(), self.train_data)
        if stage == 'validate' or stage is None:
            self.val_data = self._setup_stage(
                self.problem.get_val_conditions(), self.val_data)

    def _setup_stage(self, conditions, data):
        for cn in data:
            if isinstance(data[cn], dict):
                # only input data is given
                data[cn] = self._setup_input_data(data[cn],
                                                  conditions[cn].track_gradients)
            else:
                # pairs of inputs and targets are given
                data_dic, target = data[cn]
                data_dic = self._setup_input_data(data_dic,
                                                  conditions[cn].track_gradients)
                target = self._setup_target_data(target)
                data[cn] = data_dic, target
        return data

    def _setup_target_data(self, target):
        device = 'cuda' if self.trainer.on_gpu else 'cpu'
        if isinstance(target, torch.Tensor):
            target = target.to(device)
        else:
            target = torch.from_numpy(target).to(device)
        return target

    def _setup_input_data(self, data, track_gradients):
        device = 'cuda' if self.trainer.on_gpu else 'cpu'
        for vn in data:
            if isinstance(data[vn], torch.Tensor):
                data[vn] = data[vn].to(device)
            else:
                data[vn] = torch.from_numpy(data[vn]).to(device)

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
            batch_size=None
        )
        return dataloader

    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.val_data, 1),
            batch_size=None
        )
        return dataloader
