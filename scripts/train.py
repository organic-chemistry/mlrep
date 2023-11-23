from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from mlrep.LightModule import LightMod
from mlrep.models.CNN import VariableCNN1D
from mlrep.models.FCNN import FCNN

from mlrep.dataset_iterator import FromDataDataset
from mlrep.dataset_tools import MyIterableDataset
from mlrep.LightDataModule import DataMod
from mlrep.predict import predict

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor



parser = ArgumentParser()
parser.add_argument("--data", type=str, nargs='+',default=[])
parser.add_argument("--window_size", type=int,default=101)
parser.add_argument("--batch_size", type=int,default=32)
parser.add_argument("--patience", type=int,default=4)
parser.add_argument("--layers_dim",nargs='+', type=int,default=[21, 41, 61])

parser.add_argument("--num_workers", type=int,default=1)
parser.add_argument("--max_epochs", type=int,default=10)
parser.add_argument("--default_root_dir", type=str,default="./")
parser.add_argument("--devices", type=int,default=None)
parser.add_argument("--test", action="store_true")



parser.add_argument("--loss", type=str,default="cross_entropy",choices=["mse","cross_entropy"])


parser.add_argument("--inputs",nargs='+' , type=list,default=["H3K4me1","H3K4me3","H3K27me3",
                                                 "H3K36me3","H3K9me3","H2A.Z","H3K79me2",
                                                 "H3K9ac","H3K4me2","H3K27ac","H4K20me1"])
parser.add_argument("--outputs", nargs='+', type=str,default=["initiation"])


#parser = LightMod.add_model_specific_args(parser)

args = parser.parse_args()

#Torch specific option
torch.set_float32_matmul_precision('medium')


window_size = args.window_size
list_files = args.data
inputs = args.inputs
outputs = args.outputs
batch_size = args.batch_size
patience = args.patience 
layer_dims =  args.layers_dim # List of layer dimensions
max_epochs = args.max_epochs
default_root_dir = args.default_root_dir
num_workers = args.num_workers
devices = args.devices
patience = args.patience
loss = args.loss

training_chromosomes = [f"chr{i}" for i in range(3,23)]

validation_chromosomes = ["chr2"]

result_chromosomes = [f"chr{i}" for i in range(1,23)]

if args.test:
    training_chromosomes = ["chr1"]
    validation_chromosomes = ["chr22"]  
    result_chromosomes = ["chr1","chr22"]

data = DataMod(list_files=list_files,inputs=inputs,outputs=outputs,
               training_chromosomes=training_chromosomes,
               validation_chromosomes=validation_chromosomes,
               result_chromosomes=result_chromosomes,
               window_size=window_size,
               batch_size=batch_size,
               skip_if_nan=True,pad=True,
               num_workers=num_workers)


input_size = len(inputs)  # Number of input channels
num_classes = len(outputs)  # Number of output classes


model = VariableCNN1D(input_size, num_classes, window_size=window_size, layer_dims=layer_dims)
#model = FCNN(input_size, num_classes, window_size=window_size)

print(model)

lightning_model = LightMod(model=model,loss=loss)

    #cli.trainer.fit(cli.model)

if devices != None:
    accelerator="gpu"
else:
    accelerator="cpu"
    devices=1

CSVlog = CSVLogger(args.default_root_dir)
trainer =  L.Trainer(max_epochs=max_epochs,default_root_dir=default_root_dir,
                      logger=CSVlog ,callbacks=[EarlyStopping(monitor="val_loss",mode="min",patience=patience),
                                                LearningRateMonitor(logging_interval='epoch')],
                      accelerator=accelerator,devices=devices)



trainer.fit(lightning_model,data.train_dataloader(),data.val_dataloader())



result_root = trainer.log_dir
if f'version_{trainer.logger.version}' not in result_root :  
    result_root = os.path.join(trainer.log_dir, 'lightning_logs', f'version_{trainer.logger.version}')

predict(data,lightning_model,result_root)