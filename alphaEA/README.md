## QM9

The QM9 dataset [[1]](#1) consists of about 130K molecules with 19 regression targets. The training codes are mainly followed [[2]](#2) and modified from [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py). 

### Run a Model

The script ``train_qm9.py`` is the main file for training and evaluating a MTL model on the QM9 dataset. A set of command-line arguments is provided to allow users to adjust the training parameter configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``target``: The index of target tasks.
- ``dataset_path``: The path of the QM9 dataset.
- ``bs``: The batch size of training, validation, and test data. The default value is 128.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python train_qm9.py -h
```

If you understand those command-line arguments, you can train a MTL model by running a command like this. 

```shell
python train_qm9.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --target TARGET
```

### References

<a id="1">[1]</a> Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, and Vijay Pande. MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*, 2018, 9(2): 513-530.

<a id="2">[2]</a> Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task Learning as a Bargaining Game. In *Proceedings of the 39th International Conference on Machine Learning*, 2022.
