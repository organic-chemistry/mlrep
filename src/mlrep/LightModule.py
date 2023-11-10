import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as Fun
import torch.nn as nn

class LightMod(pl.LightningModule):
    def __init__(self,
                model: nn.Module):
                #optim: Partial[Optimizer]):
        super().__init__()
        #self.save_hyperparameters()
        self.model = model

    #@staticmethod
    #def add_model_specific_args(parent_parser):
    #    parser = parent_parser.add_argument_group("LitModel")
    #    parser.add_argument("--num_features", type=int, default=8)
    #    parser.add_argument("--kernel_size", type=int, default=3)
    #    return parent_parser

    def step(self,batch,batch_idx,log):
        x, y = batch
        y_pred = self.model(x)
        #print(y.shape)
        #print(y_pred.shape)
        #raise
        loss = Fun.mse_loss(y_pred, y)
        #loss = Fun.cross_entropy(y_pred, y)
        #print(loss)
        #raise
        self.log(log, loss , on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"training_loss")
    def validation_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"validation_loss")
    def test_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"test_loss")

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters())
        return optimizer