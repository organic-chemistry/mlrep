from lightning.pytorch import LightningDataModule
from mlrep.dataset_iterator import FromDataDataset
from mlrep.dataset_tools import MyIterableDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

class DataMod(LightningDataModule):
    """
    Create train and validation set as well as the data to predict on the whole chromosome
    """

    def __init__(self,
                list_files = [],
                training_chromosomes =  [f"chr{i}" for i in range(3,23)] ,
                validation_chromosomes =  ["chr2"] ,
                result_chromosomes = [f"chr{i}" for i in range(1,23)] ,
                inputs = ["H3K4me1","H3K4me3","H3K27me3","H3K36me3","H3K9me3","H2A.Z","H3K79me2",
                         "H3K9ac","H3K4me2","H3K27ac","H4K20me1"] ,
                outputs = ["initiation"] ,
                skip_if_nan = True,
                pad = True ,
                window_size = 101,
                batch_size = 32,
                num_workers = 4
                ) -> None:
        super().__init__()

        self.inputs = inputs
        self.outputs = outputs
        self.window_size = window_size 
        self.result_chromosomes = result_chromosomes

        self.input_dim = len(self.inputs)
        self.output_dim = len(self.outputs)
        self.batch_size=batch_size


        if type(list_files) == str:
            list_files = [list_files]
        self.list_files = list_files

        dataset_generators_train = self.load_data(list_files,training_chromosomes,skip_if_nan,pad)
        dataset_generators_validation =self.load_data(list_files,validation_chromosomes,skip_if_nan,pad)

        self.data_train = DataLoader( MyIterableDataset(dataset_generators_train), batch_size=batch_size,num_workers=num_workers)
        self.data_validation = DataLoader( MyIterableDataset(dataset_generators_validation), batch_size=batch_size)


        self.get_normalisers(inputs,self.data_train)
        self.apply_normalisers(dataset_generators_train)
        self.apply_normalisers(dataset_generators_validation)


    def apply_normalisers(self,data):
        for i in range(len(data)):
            data[i].transform_input = self.input_norm
            data[i].transform_output = self.output_norm
    

    def load_data(self,list_files,chromosomes,skip_if_nan,pad):
        inputs = self.inputs
        outputs = self.outputs
        window_size = self.window_size

        dataset_generators = []
        for file in list_files:
            data = pd.read_csv(file)
            for chromosome in chromosomes:
                sub = data.chrom == chromosome
                #print(chromosome,sum(sub))
                data_set = FromDataDataset([np.array(data[sub][inputs],dtype=np.float32),
                                                                np.array(data[sub][outputs],dtype=np.float32)], 
                                                                window_size=window_size,
                                                                skip_if_nan=skip_if_nan,pad=pad)
                dataset_generators.append(data_set)
        return dataset_generators

    def get_normalisers(self,inputs,data_train):
        """
        Iterate on the data to normalise the inputs and outputs on the training loop
        """
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

        def transform_data_inputs(data,mean_e,std_e,maxi=10):
            data =  (data -mean_e[np.newaxis,:]) / std_e[np.newaxis,:]
            data[data>maxi]=maxi
            return data
        def transform_data_outputs(data,maxi):
            return data/maxi
        
        self.input_norm = lambda x: transform_data_inputs(x,mean_e.numpy(),std_e.numpy()) 
        self.output_norm = lambda x: transform_data_outputs(x,maxi.numpy().copy())

    def setup(self, stage: str) -> None:
        pass
        """
        if stage == "fit":
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate"):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))
        """


    def train_dataloader(self) -> DataLoader:
        return self.data_train

    def val_dataloader(self) -> DataLoader:
        return self.data_validation

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.random_predict)
    
    def result_dataloader(self,file : str): #-> DataLoader:

        dataset_generators_result = self.load_data([file],self.result_chromosomes,skip_if_nan=False,pad=True)
        self.apply_normalisers(dataset_generators_result)

        nch = len(self.result_chromosomes)
        return [DataLoader( MyIterableDataset(dataset_generators_result[ch:ch+1]), batch_size=self.batch_size,num_workers=1) for ch in range(nch)]

