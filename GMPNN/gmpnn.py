#!/usr/bin/env python
# coding: utf-8



# set arguments up ([argparse](https://docs.python.org/3/library/argparse.html))
from model import config
args = config.setup()
log = args.logger



# load data 
from data import data
structure, splits = data.Loader(args).load()
log.info("Train set has " + str(len(splits['train_valid']['train'])) + " batches.\n")



# load model
from model import model
GMPNN = model.GMPNN(structure, args)
GMPNN.summary(log)



# train, test model
GMPNN.learn(splits['train_valid'])
log.info("Testing the model...")
log.info(GMPNN.test(splits['test']))