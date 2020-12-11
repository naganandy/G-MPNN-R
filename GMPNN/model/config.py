'''
data parameters
data: FB-AUTO / M-FB15K
'''
data = "FB-AUTO"



'''
environmental parameters
gpu: gpu number (range: {0, 1, 2, 3, 4, 5, 6, 7})
seed: initial seed value
w: number of workers
'''
gpu = 2
seed = 42
w = 10



'''
data load parameters
b: training batch size
B: evaluation batch size
nr: negative ratio
'''
b = 128
B = 32
nr = 10



'''
model parameters
d: number of inital dimensions
h: number of hidden dimensions
drop: hidden droput
'''
d = 256
h = 128
drop = 0.2



'''
miscellaneous parameters
lr: learning rate
epochs: number of epochs
v: frequency of validation
'''
lr = 0.0005
epochs = 1000
v = 50



# log: log on file (true) or print on console (false)
log = True



import argparse
def parse():
    """
    add and parse arguments / hyperparameters
    """
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Inductive Vertex Embedding on Multi-Relational Ordered Hypergraphs")
    p.add_argument('--data', type=str, default=data, help='data name (FB-AUTO)')

    p.add_argument('--gpu', type=int, default=gpu, help='gpu number')
    p.add_argument('--seed', type=int, default=seed, help='initial seed value')
    p.add_argument('--w', type=int, default=w, help='number of workers')

    p.add_argument('--b', type=int, default=b, help='training batch size')
    p.add_argument('--B', type=int, default=B, help='evaluation batch size')
    p.add_argument('--nr', type=int, default=nr, help='negative ratio')
    
    p.add_argument('--d', type=int, default=d, help='number of initial dimensions')
    p.add_argument('--h', type=int, default=h, help='number of hidden dimensions')
    p.add_argument('--drop', type=float, default=drop, help='hidden droput')

    p.add_argument('--lr', type=float, default=lr, help='learning rate')
    p.add_argument('--epochs', type=int, default=epochs, help='number of epochs')
    p.add_argument('--v', type=int, default=v, help='frequency of validation')
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: return True
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
    for i, k in enumerate(D): args.logger.info(k + " = " + str(D[k]) + l[i]) 


    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 


    # set device (gpu/cpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda') if args.gpu != '-1' and torch.cuda.is_available() else torch.device('cpu')   

    return args