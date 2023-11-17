class DataModule(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self,
                training_chromosomes =  [f"chr{i}" for i in range(3,23)] ,
                validation_chromosomes =  ["chr2"] ,
                result_chromosomes = [f"chr{i}" for i in range(1,23)] ,
                inputs = ["H3K4me1","H3K4me3","H3K27me3","H3K36me3","H3K9me3","H2A.Z","H3K79me2",
                         "H3K9ac","H3K4me2","H3K27ac","H4K20me1"] ,
                outputs = ["initiation"] ,
                skip_if_nan = True,
                pad = True
                ) -> None:
        super().__init__()

        #####################################################
        #create the iterator by chromosome on the data
        dataset_generators_train = self.load_data(self,list_files,training_chromosomes,inputs,outputs,skip_if_nan,pad)
        dataset_generators_validation =self.load_data(self,list_files,validation_chromosomes,inputs,outputs,skip_if_nan,pad)


        data_train = DataLoader( MyIterableDataset(dataset_generators_train), batch_size=batch_size,num_workers=args.num_workers)
        data_validation = DataLoader( MyIterableDataset(dataset_generators_validation), batch_size=batch_size)


        self.get_normalisers()
     

        ################################################
        # Give the transform (by sequence)
        for i in range(len(dataset_generators_train)):
            dataset_generators_train[i].transform_input = input_norm
            dataset_generators_train[i].transform_output = output_norm

        for i in range(len(dataset_generators_validation)):
            dataset_generators_validation[i].transform_input = lambda x: transform_data_inputs(x,mean_e.numpy(),std_e.numpy())
            dataset_generators_validation[i].transform_output = lambda x: transform_data_outputs(x,maxi.numpy().copy())

                self.random_full = RandomDataset(32, 64 * 4)

    def load_data(self,list_files,chromosomes,inputs,outputs,skip_if_nan,pad):
        dataset_generators = []
        for file in list_files:
            data = pd.read_csv(file)
            for chromosome in chromosomes
                data_set = FromDataDataset([np.array(data[sub][inputs],dtype=np.float32),
                                                                np.array(data[sub][outputs],dtype=np.float32)], 
                                                                window_size=window_size,
                                                                skip_if_nan=skip_if_nan,pad=pad)
                dataset_generators(data_set)
        return dataset_generators

    def get_normalisers(self,inputs)
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
        if stage == "fit":
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate"):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.random_train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.random_val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.random_predict)