.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/mlrep.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/mlrep
    .. image:: https://readthedocs.org/projects/mlrep/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://mlrep.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/mlrep/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/mlrep
    .. image:: https://img.shields.io/pypi/v/mlrep.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/mlrep/
    .. image:: https://img.shields.io/conda/vn/conda-forge/mlrep.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/mlrep
    .. image:: https://pepy.tech/badge/mlrep/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/mlrep
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/mlrep

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
mlrep
=====



    predicting replication using machine learning


A longer description of your project goes here...

=====
Install
=====
#version 1.9 of lightning due to parsing issues with higher versions 

```
mamba create -n mlrep pytorch-lightning=1.9 pytorch tqdm pandas pytest pytest-cov
mamba activate mlrep
pip install -e ./

```


.. _pyscaffold-notes:

=====
Training the model
=====
```
python scripts/train.py --max_epochs 100 --patienc 5 --window_size 21 --batch_size 128 --data data/K562_2000_merged_histones_init.csv.gz  --default_root_dir test/
```
It with create a log directory in test/lightning_logs/version_1/ with version changing each time you run it.
Inside you can find the metrics, the parameters as well as the checkpoints and the prediction of the model
There is an example `notebook <notebook/check_results.ipynb>` to view the results and the loss Notebook Link 


Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
