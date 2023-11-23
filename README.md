

# mlrep




    predicting replication using machine learning


A longer description of your project goes here...


## Install

version 1.9 of lightning due to parsing issues with higher versions 

```
mamba create -n mlrep pytorch-lightning pytorch tqdm pandas pytest pytest-cov
mamba activate mlrep
pip install lightning[pytorch-extra]
git clone https://github.com/organic-chemistry/mlrep.git
cd mlrep
pip install -e ./
```
For cuda support see:
https://pytorch.org/
```
mamba create -n mlrep_last pytorch-lightning pytorch tqdm pandas pytest pytest-cov  pytorch-cuda -c pytorch -c nvidi
mamba activate mlrep
pip install lightning[pytorch-extra]
git clone https://github.com/organic-chemistry/mlrep.git
cd mlrep
pip install -e ./

```


## Training the model

```
python scripts/train.py --layers_dim 10 20 40 --max_epochs 100 --patience 5 --window_size 21 --batch_size 128 --data data/K562_2000_merged_histones_init.csv.gz  --default_root_dir test/ --num_workers 1
```
It with create a log directory in test/lightning_logs/version_1/ with version changing each time you run it.
Inside you can find the metrics, the parameters as well as the checkpoints and the prediction of the model
There is an example [notebook](notebook/check_results.ipynb) to view the results and the loss Notebook Link 

On gpu:
```
python scripts/train.py --devices 1 --layers_dim 10 20 40 --patience 5 --max_epochs 100 --window_size 101 --batch_size 128 --data data/K562_2000_merged_histones_init.csv.gz  --default_root_dir test/ --num_workers 8
```

## Running the test
I implemented some test for the iterators
You can run them like that:
```
pytest
```

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
