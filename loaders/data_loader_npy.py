import numpy as np
import time
import random
import torch
from torch.utils.data import DataLoader,Dataset
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, X, y, sofa_score):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)
        self.sofa_score = torch.tensor(sofa_score, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item],self.y[item],self.sofa_score[item]


def load_data(data_name, batch_size):
    start =  time.time()
    dict_data = np.load(f'data/{data_name}.npy', allow_pickle=True).item()
    print('--------------------------------')
    print('loaded npy data')
    X_train, y_train, sofa_train = dict_data['train']['X'], dict_data['train']['y'], dict_data['train']['sofa']
    X_val, y_val, sofa_val = dict_data['val']['X'], dict_data['val']['y'], dict_data['val']['sofa']
    X_test, y_test, sofa_test = dict_data['test']['X'], dict_data['test']['y'], dict_data['test']['sofa']

    print(f'Train Set: X - {X_train.shape}, y - {y_train.shape}, sofa_system - {sofa_train.shape}, sepsis record - {np.sum(y_train)}')
    print(f'Val Set: X - {X_val.shape}, y - {y_val.shape}, sofa_system - {sofa_val.shape}, sepsis record - {np.sum(y_val)}')
    print(f'Test Set: X - {X_test.shape}, y - {y_test.shape}, sofa_system - {sofa_test.shape}, sepsis record - {np.sum(y_test)}')
    print(f'Load data, spend {time.time() - start}')

    train_dataset = MyDataset(dict_data['train']['X'], dict_data['train']['y'], dict_data['train']['sofa'])
    val_dataset = MyDataset(dict_data['val']['X'], dict_data['val']['y'], dict_data['val']['sofa'])
    test_dataset = MyDataset(dict_data['test']['X'], dict_data['test']['y'], dict_data['test']['sofa'])

    train_dataloader= DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_dataloader= DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader



def load_tensor(source, target):
    start =  time.time()
    dict_source = np.load(f'data/{source}.npy', allow_pickle=True).item()
    dict_target = np.load(f'data/{target}.npy', allow_pickle=True).item()

    def convert_to_tensor(X, y, sofa_score):
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        sofa_score = torch.tensor(sofa_score, dtype=torch.long)
        return X, y, sofa_score
    
    for key in dict_source:
        X, y, sofa_score = dict_source[key]['X'], dict_source[key]['y'], dict_source[key]['sofa']
        X, y, sofa_score = convert_to_tensor(X, y, sofa_score)
        dict_source[key]['X'] = X
        dict_source[key]['y'] = y
        dict_source[key]['sofa'] = sofa_score
    
    for key in dict_target:
        X, y, sofa_score = dict_target[key]['X'], dict_target[key]['y'], dict_target[key]['sofa']
        X, y, sofa_score = convert_to_tensor(X, y, sofa_score)
        dict_target[key]['X'] = X
        dict_target[key]['y'] = y
        dict_target[key]['sofa'] = sofa_score

    print(f'Load data, spend {time.time() - start}')
    return dict_source, dict_target
