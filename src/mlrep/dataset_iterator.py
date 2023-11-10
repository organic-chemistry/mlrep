import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from mlrep.dataset_tools import RandomFileIterator,MyIterableDataset,get_padded_view


class GenomeDataset:
    """
    Iterate in an overlapping fashion over the dataset. 
    The data is cutted by chromosomes so that there is no windows at the border between two chromosomes.
    If skip_if_nan, data with a nan will be skipped. It should be true for training and false when wanted
    to get the correct alligment when applying the model.
    inputs specify the various inputs of the model and outputs what is fitted.
    I think it is better to give only one chromosome at a time in chr_list, as when combined with the
    randomiterator it will randomly cicle throug all the chromosomes.
    """
    def __init__(self,window_size=5,skip_if_nan=True,pad=False,pad_value=0,
                 transform_input=lambda x:x, transform_output=lambda x:x):
        self.window_size = window_size
        self.skip_if_nan = skip_if_nan
        self.pad=pad
        self.pad_length = self.window_size // 2
        self.pad_value=pad_value
        self.transform_input = transform_input
        self.transform_output = transform_output
        assert(self.window_size % 2 == 1)

    def load_and_tokenize(self):
        print("To be implemented")

    def view(self,data,index):
        if not self.pad:
            return data[index[0]:index[1]]
        else:
            return get_padded_view(data,index,self.pad_length,pad_value=0)

    def create_pairs(self, chunck):
        inputs,outputs = chunck
        if self.pad:
            extra = 0
        else:
            extra = self.window_size

        for i in range(len(inputs) - extra):
            input_seq = self.view(inputs,[i,i + self.window_size])
            target_output = outputs[i + extra//2]
            if np.sum(np.isnan(target_output)) !=0 and self.skip_if_nan:
                continue
            yield self.transform_input(input_seq), self.transform_output(target_output)

    def __iter__(self):
        tokenized_text_generator = self.load_and_tokenize() #self.file_path)
        for chunck in tokenized_text_generator:
            for input_seq, target_output in self.create_pairs(chunck):
                yield input_seq, target_output

    def __len__(self):
        return sum(1 for _ in self)

class FromFileDataset(GenomeDataset):
    """
    Data loader that need a csv as well as the chromosome list the list of inputs and outputs
    """
    def __init__(self, file_path, window_size=5,skip_if_nan=True,pad=False,pad_value=0,inputs=[],outputs=[],
                 chr_list=None,dtype=np.float32):
        super().__init__(window_size=window_size,skip_if_nan=skip_if_nan,pad=pad,pad_value=pad_value)
        self.file_path = file_path
        self.chr_list = chr_list
        self.inputs = inputs
        self.outputs = outputs
        self.dtype= dtype

    def load_and_tokenize(self):
        data = pd.read_csv(self.file_path)
        #print(set(data.chrom))
        if self.chr_list is not None:
            sub = [d in self.chr_list for d in data.chrom]
            data = data[sub]
        else:
            self.chr_list = list(set(data.chrom))
        #print(self.chr_list,len(data))
        for ch in self.chr_list:
            # Here could come a normalising procedure
            sub =  data.chrom == ch
            #print(ch,sum(sub))
            yield (np.array(data[sub][self.inputs],dtype=self.dtype), 
                  np.array(data[self.outputs],dtype=self.dtype))  #Keep the yield in case we open the data iteratively

class FromDataDataset(GenomeDataset):
    """
    Iterate directly on the data
    """
    def __init__(self, data, window_size=5,skip_if_nan=True,pad=False,pad_value=0):
        super().__init__(window_size=window_size,skip_if_nan=skip_if_nan,pad=pad,pad_value=pad_value)
        self.data = data

    def load_and_tokenize(self):
        yield self.data[0],self.data[1]  #Keep the 

if __name__ == "__main__":
#text_dataset_generators,)


# Specify the directory where your data files are stored
    data_dir = '/home/jarbona/mlrep/data/'

    # Create a list of file paths
    list_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    print(list_files)
    # Create a list of tokenized generators for each file
    inputs = ["H3K4me1","H3K4me3","H3K27me3","H3K36me3","H3K9me3","H2A.Z","H3K79me2","H3K9ac","H3K4me2","H3K27ac","H4K20me1"]
    outputs = ["initiation"]
    training_chromosomes = []

    text_dataset_generators = []
    training_chromosomes = [f"chr{i}" for i in range(1,23)]

    """
    #Not fast by chromosome because of the need to open several times the initial file
    for file in list_files:
        for chromosome in training_chromosomes:
            text_dataset_generators.append(FromFileDataset(file, window_size=101,skip_if_nan=True,
                                                inputs=inputs,outputs=outputs,chr_list=[chromosome]))
    """
    for file in list_files:
        data = pd.read_csv(file)
        for chromosome in training_chromosomes:
            sub = data.chrom == chromosome
            text_dataset_generators.append(FromDataDataset([np.array(data[sub][inputs]),
                                                            np.array(data[sub][outputs])], 
                                                            window_size=3,skip_if_nan=True,pad=True))

    IteData = MyIterableDataset(text_dataset_generators)

    max_length = None  # Specify the maximum number of data points to generate

    batch_size = 32
    data_loader = DataLoader(IteData, batch_size=batch_size)

    # Iterate through the data loader
    print("First")
    import tqdm
    for input_seq,output_seq in tqdm.tqdm(data_loader):
        #print(input_seq,output_seq)
        pass
    print("Second")
    for input_seq,output_seq in tqdm.tqdm(data_loader):
        #print(input_seq.shape,output_seq)
        pass
    print("Third")
#for input_seq,output_seq in data_loader:
#    print(input_seq,output_seq)
