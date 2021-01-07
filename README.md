# NHP: Neural Hypergraph Link Prediction

[![Conference](http://img.shields.io/badge/CIKM-2020-4b44ce.svg)](https://www.cikm2020.org/) 

Source code for [CIKM 2020](https://www.cikm2020.org/) paper: **NHP: Neural Hypergraph Link Prediction**

![](./resources/model.png)

**Overview of HyperGCN:** *Given a (directed) hypergraph and node features, Neural Hyperlink Predictor (NHP) uses a hyperlink-aware graph convolutional layer and a maxmin scoring layer to score existing hyperlinks higher than non-existing vertex sets. *


### Notes:
- Please use requirements.txt to install dependencies.
- All datasets used in the paper are included in the data directory
- Codes for preprocessing new datasets / creating train-test splits are in the data/preprocess directory

### Model Usage:

- To start training and testing run:

  ```shell
  python gmpnn.py --data MFB-IND --agg max --log True
  ```

  - `--data` denotes the dataset to use
  - `--score` denotes the type of scoring (maxmin / mean)
  - `--split` is the train-test split number
  

### Citation:

```bibtex
@inproceedings{nhp_cikm20,
title = {NHP: Neural Hypergraph Link Prediction},
author = {Yadati, Naganand and and Nitin, Vikram Nimishakavi, Madhav and Yadav, Prateek and Louis, Anand and Talukdar, Partha},
booktitle = {Proceedings of the Conference on Information and Knowledge Management (CIKM)},
year = {2020}
}
```
