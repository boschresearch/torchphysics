import pytorch_lightning as pl
import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, iterations):
        super().__init__()
        self.data = data
        self.epoch_len = iterations

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        return self.data


class ProblemDataModule(pl.LightningDataModule):
    def __init__(self, problem, n_iterations):
        super().__init__()
        self.problem = problem
        self.n_iterations = n_iterations

    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        self.data = {}
        conditions = self.problem.get_train_conditions()

        for name in conditions:
            self.data[name] = conditions[name].get_data()

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        for cn in self.data:
            if isinstance(self.data[cn], dict):
                for vn in self.data[cn]:
                    self.data[cn][vn] = torch.from_numpy(self.data[cn][vn]).cuda()
                    self.data[cn][vn].requires_grad = True
                    #TODO: only set requieres_grad = True for the variables that need a
                    #      derivative
            else:
                data_dic, target = self.data[cn]
                for vn in data_dic:
                    if isinstance(data_dic[vn], torch.Tensor):
                        data_dic[vn] = data_dic[vn].cuda() #TODO: only call cuda if trainer.on_gpu is True!
                    else:
                        self.data[cn][vn] = torch.from_numpy(self.data[cn][vn]).cuda()
                if isinstance(target, torch.Tensor):
                    target = target.cuda()
                else:
                    target = torch.from_numpy(target).cuda()
                self.data[cn] = data_dic, target

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.data, self.n_iterations),
            batch_size=None
        )
        return dataloader

    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.data, self.n_iterations),
            batch_size=None
        )
        return dataloader

    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        return None
