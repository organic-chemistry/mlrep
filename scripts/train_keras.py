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

#M anging the data
# Specify the directory where your data files are stored
window_size = args.window_size
list_files = args.data

# Create a list of tokenized generators for each file
inputs = args.inputs
outputs = args.outputs
batch_size = args.batch_size



training_chromosomes = [f"chr{i}" for i in range(3,23)]
training_chromosomes = ["chr3"]
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

n_inputs = len(inputs)
mean_e = torch.zeros(n_inputs)
std_e = torch.zeros(len(inputs))
n = 0
for d in data_train:
    inp,out=d
    mean_e = (n*mean_e + torch.mean(inp.view(-1,n_inputs),0)) / (n+1)
    std_e = (n*std_e + torch.std(inp.view(-1,n_inputs)-mean_e,0)) / (n+1)
    n+=1
    #print(mean_e)
print(mean_e)
print(std_e)

def transform_data(data,mean_e,std_e):
    data =  (data -mean_e[None,:].numpy()) / std_e[None,:].numpy()
    #data[data>5]=5
    return data


for i in range(len(dataset_generators_train)):
    dataset_generators_train[i].data[0] = transform_data(dataset_generators_train[i].data[0],mean_e,std_e)
for i in range(len(dataset_generators_validation)):
    dataset_generators_validation[i].data[0] =  transform_data(dataset_generators_validation[i].data[0],mean_e,std_e)

    #print()
data_train = DataLoader( MyIterableDataset(dataset_generators_train), batch_size=batch_size)
mean_e = torch.zeros(n_inputs)
std_e = torch.zeros(len(inputs))
n = 0
for d in data_train:
    inp,out=d
    mean_e = (n*mean_e + torch.mean(inp.view(-1,n_inputs),0)) / (n+1)
    std_e = (n*std_e + torch.std(inp.view(-1,n_inputs)-mean_e,0)) / (n+1)
    n+=1
    #print(mean_e)
print(mean_e)
print(std_e)

data_validation = DataLoader( MyIterableDataset(dataset_generators_validation), batch_size=batch_size)


##########################################
# create the model
input_size = len(inputs)  # Number of input channels
num_classes = len(outputs)  # Number of output classes
layer_dims = [16, 18, 20]  # List of layer dimensions

#model = VariableCNN1D(input_size, num_classes, window_size=window_size, layer_dims=layer_dims)

from torch.utils.data import DataLoader, TensorDataset
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Assuming you have a PyTorch DataLoader 'train_loader' containing your data

# Generate sample data (replace this with your actual data loading process)
# Example assumes you have a DataLoader with 'X_train' and 'y_train'
# Replace this block with the PyTorch DataLoader usage for your case
# Make sure the data is properly formatted and loaded into PyTorch DataLoader

# Example random data


# Define a simple dense neural network in Keras
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])

# Train the model using the data from the PyTorch DataLoader

data = []
labels = []
for batch in data_train:
    if batch[0].shape[0] != args.batch_size:
        continue
    data.append(batch[0].numpy())  # Convert PyTorch tensor to NumPy
    labels.append(batch[1].numpy())  # Convert PyTorch tensor to NumPy

# Merge batches into a single NumPy array
data = np.vstack(data)
labels = np.vstack(labels)

print(data.shape,labels.shape)
# Shuffle the merged data
#indices = np.random.permutation(len(data))
#data = data[indices]
#labels = labels[indices]

model.fit(data, labels, epochs=10, batch_size=32)
"""
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(data_train)):
        # Convert PyTorch tensors to numpy arrays
        data_np = data.numpy().reshape(data.shape[0],-1)
        target_np =target.numpy()  # tf.convert_to_tensor(

        # Train the Keras model using the current batch
        model.train_on_batch(data_np, target_np)
    val_loss=0
    n=0
    for batch_idx, (data, target) in enumerate(data_validation):
        X_val = data.numpy().reshape(data.shape[0],-1)
        y_val = target.numpy()

        val_loss_t, val_acc = model.test_on_batch(X_val, y_val)
        val_loss += val_loss_t
        n+=1
    val_loss /= n
    print(f"Epoch {epoch+1}/{num_epochs} - val_loss: {val_loss} - val_acc: {val_acc}")
"""