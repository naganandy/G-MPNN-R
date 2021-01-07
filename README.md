# Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs

[![Conference](http://img.shields.io/badge/NeurIPS-2020-4b44ce.svg)](https://nips.cc/) 

Source code for NeurIPS 2020 paper: **Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs**



### Model Usage:

- Please use requirements.txt to install dependencies. To start training, run the following in GMPNN or MPNN folder.

  ```shell
  python gmpnn.py --data WP-IND --agg max --log False
  python mpnnr.py --data cora --split 1 --log False

  ```

  - `--data` denotes the dataset to use
  - `--log` indicates whether to log results (and dump checkpoints)
  - `--agg` denotes the type of aggregation for GMPNN (max / mean / sum)
  - `--split` is the split number to use for MPNNR

  

### Citation:

```bibtex
@incollection{gmpnnr_neurips20,
author = {Naganand Yadati},
title = {Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 33},
year = {2020},
publisher = {Curran Associates, Inc.}
}
```