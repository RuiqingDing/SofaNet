import time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SofaNet_local import SofaNet
from model.SofaNet_concat import SofaNet_concat
from utils.local_utils import test_with_modelfile, evaluate, train
from utils.result_utils import print_metrics_binary
from loaders.data_loader_npy import load_data
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Finetune after sharing parameters and ')
parser.add_argument('--data1', type = str, default = 'challenge_1')
parser.add_argument('--data2', type = str, default = 'mimic_1')
parser.add_argument('--feas_num', type = int, default = 27) # number of features 
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--l2', type = float, default = 0.001)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--iter', type = int, default = 10000)
parser.add_argument('--alpha', type = float, default = 0.5) # weight of each channel
parser.add_argument('--lambada', type = float, default = 0.5) # weight of mmd loss
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--only_test', action='store_true', default=False)
opt = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(opt.seed) # numpy
random.seed(opt.seed) 
torch.manual_seed(opt.seed) # cpu
torch.cuda.manual_seed_all(opt.seed) #gpu
torch.backends.cudnn.deterministic = True # cudnn

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)

def main():
    model_name = f'sofanet_{opt.data1}_{opt.data2}_{opt.batch_size}_{opt.lr}_{opt.l2}_{opt.seed}'
    model_rename = f'{opt.data2}_sofanet_{opt.data1}_{opt.data2}_{opt.batch_size}_{opt.lr}_{opt.seed}'

    model = SofaNet(input_dim=opt.feas_num, hidden_dim=128, output_dim=2, num_layers=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(f'save_dict/{model_name}')['net'])
    except:
        model.load_state_dict(torch.load(f'save_dict/{opt.mmd}_{opt.data2}_{opt.data1}_{opt.batch_size}_{opt.lr}_{opt.l2}_{opt.seed}')['net'])

    model_cur = SofaNet_concat(input_dim=opt.feas_num, hidden_dim=128, output_dim=2, num_layers=1).to(DEVICE)

    gru_names = ['gru1', 'gru2', 'gru3', 'gru4']
    para_names = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']

    for gru in gru_names:
        for para in para_names:
            model_cur.state_dict()[f"encoder2.{gru}.{para}"].copy_(model.state_dict()[f"{gru}.{para}"])
    
    
    for p in model_cur.encoder2.gru1.parameters():
        p.requires_grad = False
    for p in model_cur.encoder2.gru2.parameters():
        p.requires_grad = False
    for p in model_cur.encoder2.gru3.parameters():
        p.requires_grad = False
    for p in model_cur.encoder2.gru4.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_cur.parameters()), lr=opt.lr, weight_decay=opt.l2)

    print(f'\nStart data2({opt.data2}) Training...')
    train_dataloader, val_dataloader, test_dataloader = load_data(opt.data2, opt.batch_size)
    train(model_cur, train_dataloader, val_dataloader, test_dataloader, file_name=f'save_dict/{model_rename}', optimizer=optimizer)
                    
main()
