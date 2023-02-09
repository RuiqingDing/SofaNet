import time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mmd_loss import MMDLoss
from model.SofaNet_local import SofaNet
from utils.local_utils import test_with_modelfile, evaluate
from utils.result_utils import print_metrics_binary
from loaders.data_loader_npy import load_tensor
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Local model training')
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
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)

def sample_data(dict_data, max_idx, batch_size):
    idx_picked = random.sample(range(max_idx), batch_size)
    X = dict_data['train']['X'][idx_picked, :, :]
    y = dict_data['train']['y'][idx_picked]
    sofa = dict_data['train']['sofa'][idx_picked, 1:5] # 去掉呼吸那个维度
    X, y, sofa =  X.to(DEVICE), y.to(DEVICE), sofa.to(DEVICE)
    return X, y, sofa


def train(model, dict_data1, dict_data2, file_name, optimizer):
    history = []

    model.train()
    MMD = MMDLoss()

    val_best_loss = float('inf')
    max_auc = 0
    last_improve = 0  # 记录上次验证集loss下降的epoch数
    data1_train_num = dict_data1['train']['X'].size()[0]
    data2_train_num = dict_data2['train']['X'].size()[0]

    
    train_loss = 0
    train_acc_data1 = 0
    train_acc_data2 = 0

    for iter in range(1, opt.iter):
        x_data1, y_data1, sofa_data1 = sample_data(dict_data1, data1_train_num, opt.batch_size)
        x_data2, y_data2, sofa_data2 = sample_data(dict_data2, data2_train_num, opt.batch_size)

        loss = 0

        outputs_data1 = model(x_data1)
        outputs_data2 = model(x_data2)

        loss_data1 = F.cross_entropy(outputs_data1[-1], y_data1)
        predic_data1 = torch.max(outputs_data1[-1].data, 1)[1].cpu()
        for i in range(4):
            loss_data1 += opt.alpha * F.cross_entropy(outputs_data1[i], sofa_data1[:,i])

        loss_data2 = F.cross_entropy(outputs_data2[-1], y_data2)
        predic_data2 = torch.max(outputs_data2[-1].data, 1)[1].cpu()
        for i in range(4):
            loss_data2 += opt.alpha * F.cross_entropy(outputs_data2[i], sofa_data2[:,i])

        mmd_loss = MMD(source= outputs_data1[4], target = outputs_data2[4])
        loss += loss_data1 + loss_data2 + opt.lambada * mmd_loss
        # print(loss.cpu().detach().numpy(), loss_data1.cpu().detach().numpy(), loss_data2.cpu().detach().numpy(), mmd_loss.cpu().detach().numpy())

        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_data1 = y_data1.data.cpu()
        true_data2 = y_data2.data.cpu()

        train_acc_data1 += metrics.accuracy_score(true_data1, predic_data1)
        train_acc_data2 += metrics.accuracy_score(true_data2, predic_data2)

        if (iter) % 100 == 0: 
            train_acc_data1_cur = train_acc_data1 / iter
            train_acc_data2_cur = train_acc_data2 / iter
            train_loss_cur = train_loss / iter
            val_loss, val_acc_data1, val_acc_data2, ret = evaluate(model, dict_data1, dict_data2)
            history.append(ret)

            val_auc = (ret['data1']['auroc'] + ret['data2']['auroc'])/2 # 2个数据的AUC均值

            if val_loss < val_best_loss:
            # if val_auc > max_auc:
                last_improve = iter
                val_best_loss = val_loss
                max_auc = val_auc
                state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter
                }
                torch.save(state, file_name)
                # print(f'\n------------Save best model on Iteration {iter}------------\n')
                improve = '*'
            else:
                improve = ''

            print('Iter {0:>4}, train_loss: {1:>5.3}, train_acc_data1: {2:>6.2%}, train_acc_data2: {3:>6.2%},\n\t   val_loss: {4:>5.3}, val_acc_data1: {5:>6.2%} , val_acc_data2: {6:>6.2%} {7}'.format(iter, train_loss_cur, train_acc_data1_cur, train_acc_data2_cur, val_loss, val_acc_data1, val_acc_data2, improve))

            if iter - last_improve > 2000:
                # 早停
                print("No optimization for a long time, auto-stopping...")
                break

        model.train()

    # Finish training and test
    # use the best model
    test_with_modelfile(model, file_name, dict_data1, dict_data2)


def evaluate_single_domain(model, dict_data, test = False):
    model.eval()

    if test:
        datas, labels, sofa_scores = dict_data['test']['X'], dict_data['test']['y'], dict_data['test']['sofa']
    else:
        datas, labels, sofa_scores = dict_data['val']['X'], dict_data['val']['y'], dict_data['val']['sofa']
    
    with torch.no_grad():
        sofa_scores= sofa_scores[:, 1:5] # 去掉呼吸那个维度
        datas, labels, sofa_scores = datas.to(DEVICE), labels.to(DEVICE), sofa_scores.to(DEVICE)
        outputs = model(datas)
        loss =  F.cross_entropy(outputs[-1], labels)
        probability = torch.softmax(outputs[-1], dim=1)
        for i in range(4):
            loss += opt.alpha * F.cross_entropy(outputs[i], sofa_scores[:,i])

        labels = labels.data.cpu().numpy()
        probability = np.round(probability.cpu().detach().numpy(), 4)
    
    probability = np.array(probability)
    predict = np.argmax(probability, axis=1)
    acc = metrics.accuracy_score(labels, predict)


    ret = print_metrics_binary(labels, predict, probability, verbose=0)
    return acc, loss, ret, outputs[4]


def evaluate(model, dict_data1, dict_data2, test=False):
    # MMD = MMDLoss()
    acc_data1, loss_data1, ret_data1, embed_data1 = evaluate_single_domain(model, dict_data1, test=test)
    acc_data2, loss_data2, ret_data2, embed_data2 = evaluate_single_domain(model, dict_data2, test=test)
    # mmd_loss = MMD(data1=embed_data1, data2=embed_data2)
    # loss_all = loss_data1 + loss_data2 + opt.lambada * mmd_loss
    loss_all = loss_data1 + loss_data2 
    ret = {'data1': ret_data1, 'data2': ret_data2}
    if test:
        return ret
    else:
        return loss_all, acc_data1, acc_data2, ret


def test_with_modelfile(model, model_file, dict_data1, dict_data2):
    # use the saved model to get test results
    model.load_state_dict(torch.load(model_file)['net'])
    test(model, dict_data1, dict_data2)


def test(model, dict_data1, dict_data2):
    # test
    model.eval()
    ret_all = evaluate(model, dict_data1, dict_data2, test=True)
    print('############### Result...')
    for domain in ret_all:
        ret = ret_all[domain]
        print('DOMAIN: %s ----------------' % domain)
        print("accuracy = {}".format(ret['acc']))
        print("AUC of ROC = {}".format(ret['auroc']))
        print("AUC of PRC = {}".format(ret['auprc']))
        print("min(+P, Se) = {}".format(ret['minpse']))
        print("f1_score = {}".format(ret['f1_score']))
        print("Confusion matrix =\n{}".format(ret['confusion_matrix']))


def main():
    model_name = f'sofanet_{opt.data1}_{opt.data2}_{opt.batch_size}_{opt.lr}_{opt.l2}_{opt.seed}'

    dict_data1, dict_data2 = load_tensor(opt.data1, opt.data2)

    model = SofaNet(input_dim=opt.feas_num, hidden_dim=128, output_dim=2, num_layers=1, return_embedding=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    if opt.only_test: # if only test
        test_with_modelfile(model, f'save_dict/{model_name}', dict_data1, dict_data2)
    else: # train model
        print('\nStart Training...')
        train(model, dict_data1, dict_data2, file_name=f'save_dict/{model_name}', optimizer=optimizer)

main()