'''
data parameters
data: cora / dblp / arXiv / acm
split: train-test split used for the dataset
'''
data = "dblp"
split = 2



'''
model parameters
h: number of hidden dimensions
drop: hidden droput
relu: flag for relu non-linearity
'''
h = 1024
drop = 0.0
relu = False



'''
miscellaneous parameters
lr: learning rate
epochs: number of epochs
decay: weight decay
'''
lr = 0.001
epochs = 200
decay = 0.0005



'''
environmental parameters
gpu: gpu number (range: {0, 1, 2, 3, 4, 5, 6, 7})
seed: initial seed value
log: log on file (true) or print on console (false)
'''
gpu = 0
seed = 42
log = False



import argparse
def parse():
    """
    add and parse arguments / hyperparameters
    """
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Inductive Vertex Embedding on Multi-Relational Ordered Hypergraphs")

    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: return True

    p.add_argument('--data', type=str, default=data, help='data name (FB-AUTO)')
    p.add_argument('--split', type=str, default=split, help='train-test split used for the dataset')

    p.add_argument('--h', type=int, default=h, help='number of hidden dimensions')
    p.add_argument('--drop', type=float, default=drop, help='hidden droput')
    p.add_argument("--relu", default=relu, type=str2bool, help="flag for relu non-linearity")

    p.add_argument('--lr', type=float, default=lr, help='learning rate')
    p.add_argument('--epochs', type=int, default=epochs, help='number of epochs')
    p.add_argument('--decay', type=float, default=decay, help='weight decay')
    
    p.add_argument('--gpu', type=int, default=gpu, help='gpu number')
    p.add_argument('--seed', type=int, default=seed, help='initial seed value')
    p.add_argument("--log", default=log, type=str2bool, help="log on file (true) or print on console (false)")
    
    p.add_argument('-f') # for jupyter default
    return p.parse_args()



import os, inspect, logging
class Logger():
    def __init__(self, args):
        '''
        Initialise logger 
        '''
        # setup checkpoint directory
        current = os.path.abspath(inspect.getfile(inspect.currentframe()))
        Dir = os.path.join(os.path.split(os.path.split(current)[0])[0], "checkpoints")
        self.log = args.log
        
        # setup log file  
        if args.log: 
            if not os.path.exists(Dir): os.makedirs(Dir)
            name = str(len(os.listdir(Dir)) + 1)
            
            Dir = os.path.join(Dir, name)
            if not os.path.exists(Dir): os.makedirs(Dir)
            args.dir = Dir

            # setup logging
            logger = logging.getLogger(__name__)
            
            file = os.path.join(Dir, name + ".log")
            logging.basicConfig(format="%(asctime)s - %(levelname)s -   %(message)s", filename=file, level=logging.INFO)
            self.logger = logger
        

    def info(self, s):
        if self.log: self.logger.info(s)
        else: print(s)



import torch, numpy as np
def setup():
    
    # parse arguments
    args = parse()
    args.logger = Logger(args)
    D = vars(args)


    # log configuration
    l = ['']*(len(D)-1) + ['\n\n']
    args.logger.info("Arguments are as follows")
    for i, k in enumerate(D): args.logger.info(k + ": " + str(D[k]) + l[i]) 


    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 


    # set device (gpu/cpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda') if args.gpu != '-1' and torch.cuda.is_available() else torch.device('cpu')   

    return args