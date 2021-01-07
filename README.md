# Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs

[![Conference](http://img.shields.io/badge/NeurIPS-2020-4b44ce.svg)](https://nips.cc/) 

Source code for [NeurIPS 2020](https://www.cikm2020.org/) paper: **Neural Message Passing for Multi-Relational Ordered and Recursive Hypergraphs**


### Notes:
- Please use requirements.txt to install dependencies.
- All datasets used in the paper are included in the data directory

### Model Usage:

- To start training and testing run:

  ```shell
  python gmpnn.py --data WP-IND --agg max --log True
  python mpnnr.py --data cora --split 1 --log True

  ```

  - `--data` denotes the dataset to use
  - `--log` indicates whether to log results (and dump checkpoints)
  - `--agg` denotes the type of aggregation for G-MPNN (max / mean / sum)
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
