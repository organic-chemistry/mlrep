from torch import optim
import torch.nn.functional as Fun
import torch.nn as nn
import torch
import lightning as L

class LightMod(L.LightningModule):
    def __init__(self,
                model: nn.Module,
                loss = "cross_entropy",
                patience=3):
                #optim: Partial[Optimizer]):
        super().__init__()
        #self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.patience=patience


    def step(self,batch,batch_idx,log):
        x, y = batch

        logit=False
        if self.loss == "cross_entropy":
            logit=True  # this is because the loss is more stable
        
        y_pred = self.model(x,logit=logit)
        
        if self.loss == "cross_entropy":
            loss = Fun.binary_cross_entropy_with_logits(y_pred, y)
        elif loss == "mse":
            self.loss = Fun.mse_loss(y_pred, y)

        self.log(log, loss , on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"training_loss")
    def validation_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"val_loss")
    def test_step(self, batch, batch_idx):
        return self.step(batch,batch_idx,"test_loss")

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters())
        lr_scheduler = {"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                               patience=self.patience ,
                                                                               min_lr = 0.01,factor=0.5),
                        "monitor":"val_loss"}
        return [optimizer], [lr_scheduler]