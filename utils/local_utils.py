import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from utils.result_utils import print_metrics_binary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)


def train(model, train_dataloader, val_dataloader, test_dataloader, file_name, optimizer):
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, min_lr=0.0001, patience=10)
    max_roc = 0
    history = []

    model.train()
    val_best_loss = float('inf')
    last_improve = 0 
    for epoch in range(1, 100):
        train_loss = 0
        train_acc = 0
        for i, (datas, labels, sofa_scores) in enumerate(train_dataloader):
            sofa_scores = sofa_scores[:, 1:5] # 4 channels
            datas, labels, sofa_scores = datas.to(DEVICE), labels.to(DEVICE), sofa_scores.to(DEVICE)
            outputs = model(datas)
            if type(outputs) == list: # multi channels
                loss =  F.cross_entropy(outputs[-1], labels)
                predic = torch.max(outputs[-1].data, 1)[1].cpu()
                for i in range(4):
                    loss += 0.5 * F.cross_entropy(outputs[i], sofa_scores[:,i])
            else:
                loss = F.cross_entropy(outputs, labels)
                predic = torch.max(outputs.data, 1)[1].cpu()

            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            true = labels.data.cpu()
            train_acc += metrics.accuracy_score(true, predic)

            # if (i+1) % 100 == 0:
            #     print('epoch {}, iter {}, train_loss: {}, train_acc: {}'.format(epoch, i+1, train_loss/i, train_acc/i))
            
        train_acc = train_acc / len(train_dataloader)
        train_loss = train_loss / len(train_dataloader)
        val_acc, val_loss, ret = evaluate(model, val_dataloader)
        lr_scheduler.step(val_loss)
        history.append(ret)

        model.train()
        if val_loss < val_best_loss:
            val_best_loss = val_loss
            # torch.save(model.state_dict(), save_file)
            improve = '*'
        else:
            improve = ''

        print('Epoch {0:>4}, train_loss: {1:>5.2}, train_acc: {2:>6.2%}, val_loss: {3:>5.2}, val_acc: {4:>6.2%} {5}'.format(epoch, train_loss, train_acc, val_loss, val_acc, improve))

        cur_auroc = ret['auroc']
        if cur_auroc > max_roc:
            max_roc = cur_auroc
            last_improve = epoch
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, file_name)
            print(f'\n------------Save best model on Epoch {epoch}------------\n')

        if epoch - last_improve > 20:
            print("No optimization for a long time, auto-stopping...")
            break

    # Finish training and test
    # use the best model
    # test(model, test_dataloader)
    if type(test_dataloader) == list:
        for i, test_dataloader_i in enumerate(test_dataloader):
            print(f'-------- Test {i+1} --------------')
            test_with_modelfile(model, file_name, test_dataloader_i)
    else:
        test_with_modelfile(model, file_name, test_dataloader)



def test(model, test_iter):
    # test
    model.eval()
    start_time = time.time()
    ret = evaluate(model, test_iter, test=True)
    print('Result...')
    print("accuracy = {}".format(ret['acc']))
    # print("precision class 0 = {}".format(ret['prec0']))
    # print("precision class 1 = {}".format(ret['prec1']))
    # print("recall class 0 = {}".format(ret['rec0']))
    # print("recall class 1 = {}".format(ret['rec1']))
    print("AUC of ROC = {}".format(ret['auroc']))
    print("AUC of PRC = {}".format(ret['auprc']))
    print("min(+P, Se) = {}".format(ret['minpse']))
    print("f1_score = {}".format(ret['f1_score']))
    print("Confusion matrix =\n{}".format(ret['confusion_matrix']))
    time_dif = get_time_dif(start_time)
    print("Test Time usage:", time_dif)


# in current epoch, print the train set loss/accurancy and validation set loss/accurancy
def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    labels_all = np.array([], dtype=int)
    probability_all = []
    
    with torch.no_grad():
        for (datas, labels, sofa_scores) in data_iter:
            sofa_scores= sofa_scores[:, 1:5] # 去掉呼吸那个维度
            datas, labels, sofa_scores = datas.to(DEVICE), labels.to(DEVICE), sofa_scores.to(DEVICE)
            outputs = model(datas)
            if type(outputs) == list: # multi channels
                loss =  F.cross_entropy(outputs[-1], labels)
                probability = torch.softmax(outputs[-1], dim=1)
                for i in range(4):
                    loss += F.cross_entropy(outputs[i], sofa_scores[:,i])
            else:
                loss = F.cross_entropy(outputs, labels)
                probability = torch.softmax(outputs, dim=1)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            probability = np.round(probability.cpu().detach().numpy(), 4)
            probability_all.extend(probability)
    
    probability_all = np.array(probability_all)
    predict_all = np.argmax(probability_all, axis=1)
    acc = metrics.accuracy_score(labels_all, predict_all)

    ret = print_metrics_binary(labels_all, predict_all, probability_all, verbose=0)

    if test:
        return ret
    return acc, loss_total / len(data_iter), ret



def test_with_modelfile(model, model_file, data_iter):
    # use the saved model to get test results
    model.load_state_dict(torch.load(model_file)['net'])
    test(model, data_iter)