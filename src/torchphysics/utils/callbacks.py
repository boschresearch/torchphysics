from pytorch_lightning.callbacks import Callback

class WeightSaveCallback(Callback):
    def __init__(self, check_interval=200):
        super().__init__()
        self.check_interval=check_interval

    def on_train_start(self, trainer, pl_module):
        return
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return

    def on_train_end(self, trainer, pl_module):
        return