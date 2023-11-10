from argparse import ArgumentParser
import pandas as pd
import numpy as np

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


from mlrep.LightModule import LightMod
from mlrep.models.CNN import VariableCNN1D
from mlrep.models.FCNN import FCNN

from mlrep.dataset_iterator import FromDataDataset
from mlrep.dataset_tools import MyIterableDataset


parser = ArgumentParser()
parser.add_argument("--data", type=str, nargs='+',default=[])
parser.add_argument("--window_size", type=int,default=101)
parser.add_argument("--batch_size", type=int,default=32)
parser.add_argument("--inputs",nargs='+' , type=list,default=["H3K4me1","H3K4me3","H3K27me3",
                                                 "H3K36me3","H3K9me3","H2A.Z","H3K79me2",
                                                 "H3K9ac","H3K4me2","H3K27ac","H4K20me1"])
parser.add_argument("--outputs", nargs='+', type=str,default=["initiation"])

parser = pl.Trainer.add_argparse_args(parser)

#parser = LightMod.add_model_specific_args(parser)

args = parser.parse_args()


window_size = args.window_size
list_files = args.data

# Create a list of tokenized generators for each file
inputs = args.inputs
outputs = args.outputs
batch_size = args.batch_size



training_chromosomes = [f"chr{i}" for i in range(3,23)]
validation_chromosomes = ["chr2"]


dataset_generators_train = []
dataset_generators_validation = []

for file in list_files:
    data = pd.read_csv(file)
    for chromosome in training_chromosomes+validation_chromosomes:
        sub = data.chrom == chromosome
        data_set = FromDataDataset([np.array(data[sub][inputs],dtype=np.float32),
                                                        np.array(data[sub][outputs],dtype=np.float32)], 
                                                        window_size=window_size,
                                                        skip_if_nan=True,pad=False)
        if chromosome in training_chromosomes:
            dataset_generators_train.append(data_set)
        elif chromosome in validation_chromosomes:
            dataset_generators_validation.append(data_set)


data_train = DataLoader( MyIterableDataset(dataset_generators_train), batch_size=batch_size)
data_validation = DataLoader( MyIterableDataset(dataset_generators_validation), batch_size=batch_size)


################################################
#Normalise the data
n_inputs = len(inputs)
mean_e = torch.zeros(n_inputs)
std_e = torch.zeros(len(inputs))
maxi=0
n = 0
for d in data_train:
    inp,out=d
    mean_e = (n*mean_e + torch.mean(inp.view(-1,n_inputs),0)) / (n+1)
    std_e = (n*std_e + torch.std(inp.view(-1,n_inputs)-mean_e,0)) / (n+1)
    maxi = max(maxi,torch.max(out))
    n+=1
    #print(mean_e)
print(mean_e,std_e)
print(maxi)

def transform_data_inputs(data,mean_e,std_e,maxi=10):
    data =  (data -mean_e[np.newaxis,:]) / std_e[np.newaxis,:]
    data[data>maxi]=maxi
    return data
def transform_data_outputs(data,maxi):
    return data/maxi


for i in range(len(dataset_generators_train)):
    dataset_generators_train[i].transform_input = lambda x: transform_data_inputs(x,mean_e.numpy(),std_e.numpy())
    dataset_generators_train[i].transform_output = lambda x: transform_data_outputs(x,maxi.numpy().copy())

for i in range(len(dataset_generators_validation)):
    dataset_generators_validation[i].transform_input = lambda x: transform_data_inputs(x,mean_e.numpy(),std_e.numpy())
    dataset_generators_validation[i].transform_output = lambda x: transform_data_outputs(x,maxi.numpy().copy())



##########################################
# create the model
input_size = len(inputs)  # Number of input channels
num_classes = len(outputs)  # Number of output classes
layer_dims = [16, 18, 20]  # List of layer dimensions

model = VariableCNN1D(input_size, num_classes, window_size=window_size, layer_dims=layer_dims)
#model = FCNN(input_size, num_classes, window_size=window_size)

print(model)

lightning_model = LightMod(model=model)

trainer =  pl.Trainer.from_argparse_args(args, callbacks=[EarlyStopping(monitor="validation_loss",
                                                                         mode="min",patience=4)])


trainer.fit(model=lightning_model, train_dataloaders=data_train,
                          val_dataloaders=data_validation)
