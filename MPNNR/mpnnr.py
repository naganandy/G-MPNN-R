#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# # set arguments up ([argparse](https://docs.python.org/3/library/argparse.html))

# In[2]:


from model import config
args = config.setup()
log = args.logger


# # load data 

# In[3]:


from data import data
dataset, splits = data.load(args)


# # load model

# In[4]:


from model import model
MPNNR = model.MPNNR(dataset, args)
MPNNR.summary(log)


# # train, test model
# 

# In[5]:


MPNNR.learn(splits['train'])
acc = MPNNR.Test(splits['test'])
log.info("Accuracy is " + str(round(acc.item()*100, 3)))