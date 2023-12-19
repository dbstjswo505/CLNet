import time
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from torch import nn
from torch.autograd import Variable
from utils import print_results, save_checkpoint, print_results_test
import pandas as pd

def confusion_matrix(l, o):
    matrix = np.zeros((5,5))
    L = len(l)
    for i in range(L):
        label = int(l[i])
        pred = int(o[i])
        matrix[label][pred] = matrix[label][pred] + 1
    return matrix

def one_hot_encode(index, targets=5):
    B = index.shape
    ohe = torch.zeros((B,targets),dtype=torch.long)
    for i in range(B):
        # 
        ohe[i] = 2

def save_df(df, path):
    df.to_csv(path, index=True)

def tIOU_calculator(st_list_pr, st_list_gt, ed_list_pr, ed_list_gt, dur):
    """
    st_list_pr: predicted start time list
    st_list_gt: ground truth start time list
    ed_list_pr: predicted end time list
    ed_list_gt: ground truth end time list
    dur: after this duration, the data is not considered
    """
    tIOU = 0
    for i in range(len(st_list_pr)):
        st_pr = st_list_pr[i]
        st_gt = st_list_gt[i]
        ed_pr = ed_list_pr[i]
        ed_gt = ed_list_gt[i]
        st_ed_list = [st_pr, st_gt, ed_pr, ed_gt]
        flag = 0
        
        # edge case
        if st_pr >= ed_pr and st_gt >= ed_gt:
            tIOU += 0
        
        # case 1. st_pr < st_gt < ed_pr < ed_gt
        elif st_pr <= st_gt and st_gt <= ed_pr and ed_pr <= ed_gt:
            for item in st_ed_list:
                if item > dur[i]:
                    flag = 1
                    break
                    
            if flag == 0:
                tIOU += (min(ed_pr, ed_gt) - max(st_pr, st_gt)) / (max(ed_pr, ed_gt) - min(st_pr, st_gt))
            
        # case 2. st_gt < st_pr < ed_gt < ed_pr
        elif st_gt <= st_pr and st_pr <= ed_gt and ed_gt <= ed_pr:
            for item in st_ed_list:
                if item > dur[i]:
                    flag = 1
                    break
            
            if flag == 0:
                tIOU += (min(ed_pr, ed_gt) - max(st_pr, st_gt)) / (max(ed_pr, ed_gt) - min(st_pr, st_gt))
            
        # case 3. st_pr < ed_pr < st_gt < ed_gt
        elif st_pr <= ed_pr and ed_pr <= st_gt and st_gt <= ed_gt:
            tIOU += 0
        
        # case 4. st_gt < ed_gt < st_pr < ed_pr
        elif st_gt <= ed_gt and ed_gt <= st_pr and st_pr <= ed_pr:
            tIOU += 0
        
    return tIOU / len(st_list_pr)

def tIoU_cal(predictions, labels, max_top=50):
    # prediction = [[st,ed], [], [], ... ] # B x (L x L) x 2
    # label = [st, ed] # B x 2

    # prediction : [st (float), ed (float)]
    # gt: [st (float), ed (float)]
    B,_,_ = predictions.shape
    tIoU = torch.zeros((B,max_top))
    for i in range(B):
        for j in range(max_top):
            pred = predictions[i,j] # predictions[i][j]
            gt = labels[i]
            intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
            union = (max(pred[1], gt[1]) - min(pred[0], gt[0]))
            if union == 0:
                tIoU[i,j] =  0
            else:
                tIoU[i,j] = 1.0 * intersection / union
    return tIoU
    
def rank_tIOU(st_outputs, ed_outputs, st_labels, ed_labels, dur): # tIOU_mode

    B = dur.shape[0]
    a_mask = torch.arange((900)).view(1,-1)
    a_mask = a_mask.repeat(B,1)
    dur_ = dur.view(-1,1)
    b_mask = a_mask < dur_
    #c_mask = torch.einsum('bl,bl->bll',b_mask,b_mask
    c_mask = ~b_mask
    mask = c_mask*-1000

    st_prob_ = st_outputs.cpu() + mask
    st_prob = F.softmax(st_prob_, dim=1)

    ed_prob_ = ed_outputs.cpu() + mask
    ed_prob = F.softmax(ed_prob_, dim=1)

    j_prob = torch.einsum('bn,bm->bnm', st_prob, ed_prob)

    nms_msk = torch.tensor(np.load('sparsity_mask.npy'))
    nms_msk = nms_msk.view(1,900,900)

    causal_msk = torch.tensor(np.load('causal_mask.npy'))
    causal_msk = causal_msk.view(1,900,900)

    j_prob_ = j_prob * causal_msk * nms_msk

    fltn_j_prob = j_prob_.view(B,-1)

    _, idx = fltn_j_prob.sort(descending=True) # idx = top-n prediction

    idx_st = idx//900
    idx_ed = idx%900

    _,tmp_st = st_labels.sort(descending=True, dim=1)
    _,tmp_ed = ed_labels.sort(descending=True, dim=1)
    
    a_tmp = tmp_st[:,0]
    b_tmp = tmp_ed[:,0]

    label_st_ed = torch.stack([a_tmp,b_tmp], dim=1)
    perdiction_st_ed = torch.stack([idx_st, idx_ed], dim=2)

    tIoUs = tIoU_cal(perdiction_st_ed, label_st_ed)

    # tIoU.shape = [48, 50]
    # tIoU = 0.7
    R1_07 = (tIoUs[:,:1] > 0.7).flatten()
    R5_07 = (tIoUs[:,:5] > 0.7).sum(dim=1) > 0
    R10_07 = (tIoUs[:,:10] > 0.7).sum(dim=1) > 0
    R50_07 = (tIoUs[:,:50] > 0.7).sum(dim=1) > 0

    R1_05 = (tIoUs[:,:1] > 0.5).flatten()
    R5_05 = (tIoUs[:,:5] > 0.5).sum(dim=1) > 0
    R10_05 = (tIoUs[:,:10] > 0.5).sum(dim=1) > 0
    R50_05 = (tIoUs[:,:50] > 0.5).sum(dim=1) > 0

    R1_03 = (tIoUs[:,:1] > 0.3).flatten()
    R5_03 = (tIoUs[:,:5] > 0.3).sum(dim=1) > 0
    R10_03 = (tIoUs[:,:10] > 0.3).sum(dim=1) > 0
    R50_03 = (tIoUs[:,:50] > 0.3).sum(dim=1) > 0

    return R1_07, R5_07, R10_07, R50_07, R1_05, R5_05, R10_05, R50_05, R1_03, R5_03, R10_03, R50_03
    # R1_03 = B [True, False, True, True, ..., ]

def round_converter(df_row, num):
    df_after = []
    for i in range(len(df_row)):
        df_after.append(round(df_row[i], num))
        
    return df_after

def test(net, dataloaders, model_name, criterion, phases=["test"], max_epochs=1, classlabels=None):
    

    st_criterion = nn.CrossEntropyLoss()
    ed_criterion = nn.CrossEntropyLoss()
    fl_criterion = nn.CrossEntropyLoss()
    
    results = []
    
    df = pd.DataFrame(columns=['epoch',
                               'test_st_loss', 'test_ed_loss', 
                               'test_R@1_tIOU_03', 'test_R@5_tIOU_03', 'test_R@10_tIOU_03', 'test_R@50_tIOU_03',
                               'test_R@1_tIOU_05', 'test_R@5_tIOU_05', 'test_R@10_tIOU_05', 'test_R@50_tIOU_05',
                               'test_R@1_tIOU_07', 'test_R@5_tIOU_07', 'test_R@10_tIOU_07', 'test_R@50_tIOU_07',
                               #
                               'time'])
    
    for epoch in range(max_epochs):

        results.append(dict())

        # Each epoch has a training and validation phase
        since = time.time()
        
        # R1_07, R5_07, R10_07, R50_07, R1_05, R5_05, R10_05, R50_05, R1_03, R5_03, R10_03, R50_03
        # R_accumulated = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for phase in phases:

            # since = time.time()

            if phase == 'train':
                net.train(True)
            else:
                net.eval()

            fl_running_loss, st_running_loss, ed_running_loss = 0., 0., 0.
            st_num_correct, ed_num_correct = 0., 0.
            st_total_samples, ed_total_samples = 0., 0.

            o = np.zeros((0,))
            l = np.zeros((0,))

            o_ed = np.zeros((0, ))
            l_ed = np.zeros((0, ))

            #NOTE:  _tIOU_03, _tIOU_05, _tIOU_07
            results[-1][phase] = dict(st_loss=[], ed_loss=[], st_acc=[], ed_acc=[], tIOU=[], 
                                      R1_tIOU_03=[], R5_tIOU_03=[], R10_tIOU_03=[], R50_tIOU_03=[], 
                                      R1_tIOU_05=[], R5_tIOU_05=[], R10_tIOU_05=[], R50_tIOU_05=[], 
                                      R1_tIOU_07=[], R5_tIOU_07=[], R10_tIOU_07=[], R50_tIOU_07=[], 
                                      time=[])

            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):
                fl_running_loss, st_running_loss, ed_running_loss = 0., 0., 0.
                R_accumulated = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                inputs = data[0].view(-1, 1, 900, 205)
                targets = data[1]['target'] # tensor([4, 2, 3, 3, 0, ...]) 
                targets_onehot = F.one_hot(targets, num_classes=5)
                dur = data[1]['dur']
                labels = torch.stack([data[1]['st'][0],data[1]['st'][1]], dim=1) #
                
                fl_labels = data[1]['fl_labels'].cuda()
                st_labels = data[1]['st_labels'].cuda()
                ed_labels = data[1]['ed_labels'].cuda()
                gap = data[1]['gap'].cuda()
                
                if phase != 'train':
                    with torch.no_grad():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # zero the parameter gradients
                # optimizer.zero_grad()

                fl_outputs, st_outputs, ed_outputs = net(inputs, targets_onehot, dur) # output.shape = torch.Size([64, 900, 1])
                # save for experiment
                # target: 0, 1: O, 2, 3, 4, [125, 314], dur = 496
                tmp_f = fl_outputs[9].cpu().detach().numpy()
                tmp_s = st_outputs[9].cpu().detach().numpy()
                tmp_e = ed_outputs[9].cpu().detach().numpy()
                df_f = pd.DataFrame(tmp_f)
                df_s = pd.DataFrame(tmp_f)
                df_e = pd.DataFrame(tmp_f)

                #df_f.to_csv("frame_pred.csv", index = False)
                #df_s.to_csv("st_pred.csv", index = False)
                #df_e.to_csv("ed_pred.csv", index = False)

                #NOTE: error

                fl_loss = fl_criterion(fl_outputs, fl_labels) 
                st_loss = st_criterion(st_outputs, st_labels)
                ed_loss = ed_criterion(ed_outputs, ed_labels)


                #NOTE: 
                R_list = rank_tIOU(st_outputs, ed_outputs, st_labels, ed_labels, dur)
                B = data[1]['target'].shape
                # for R_l, R_a in zip(R_list, R_accumulated):
                #     R_a += R_l.sum().item()

                for idx, R_l in enumerate(R_list):
                    R_accumulated[idx] += R_l.sum().item() / B[0]

                # statistics
                fl_running_loss += (fl_loss.item() / (sum(gap)/30))
                st_running_loss += st_loss.item()
                ed_running_loss += ed_loss.item()
                
                # Accuracy: st
                st_num_correct += st_outputs[:st_labels.size(0)].max(dim=1)[1].eq(st_labels.max(1)[1]).sum().item()
                st_total_samples += len(st_outputs)
                
                o = np.concatenate((o, st_outputs[:st_labels.size(0)].max(1)[1].cpu().data.numpy()))
                l = np.concatenate((l, st_labels.max(1)[1].cpu().data.numpy()))

                # Accuracy: ed
                ed_num_correct += ed_outputs[:ed_labels.size(0)].max(dim=1)[1].eq(ed_labels.max(1)[1]).sum().item()
                ed_total_samples += len(ed_outputs)
                o_ed = np.concatenate((o_ed, ed_outputs[:ed_labels.size(0)].max(1)[1].cpu().data.numpy())) # tensor에서 array로 변경
                l_ed = np.concatenate((l_ed, ed_labels.max(1)[1].cpu().data.numpy()))
                
                st_list_pr = st_outputs.max(1)[1].cpu().data.numpy()
                st_list_gt = st_labels.max(1)[1].cpu().data.numpy()

                ed_list_pr = ed_outputs.max(1)[1].cpu().data.numpy()
                ed_list_gt = ed_labels.max(1)[1].cpu().data.numpy()
                
                # NOTE: Calculate tIOU
                # tIOU = tIOU_calculator(st_list_pr, st_list_gt, ed_list_pr, ed_list_gt, dur)
 
                # del inputs, outputs, labels, loss
                
                results[-1][phase]["st_loss"].append(st_running_loss / (idx + 1))
                results[-1][phase]["st_acc"].append(st_num_correct / st_total_samples)
                
                results[-1][phase]["ed_loss"].append(ed_running_loss / (idx + 1))
                results[-1][phase]["ed_acc"].append(ed_num_correct / ed_total_samples)
                
                results[-1][phase]["time"].append(time.time() - since)
                #results[-1][phase]["tIOU"].append(tIOU)
                
                #
                print()

                R1_07, R5_07, R10_07, R50_07, R1_05, R5_05, R10_05, R50_05, R1_03, R5_03, R10_03, R50_03 = R_accumulated
                R_tIOU_03_list = [R1_03, R5_03, R10_03, R50_03]
                R_tIOU_05_list = [R1_05, R5_05, R10_05, R50_05]
                R_tIOU_07_list = [R1_07, R5_07, R10_07, R50_07]
                temp = ["R1", "R5", "R10", "R50"]
                for idx, item in enumerate(R_tIOU_03_list):
                    name = temp[idx] + "_tIOU_03"
                    results[-1][phase][name].append(item)
                
                for idx, item in enumerate(R_tIOU_05_list):
                    name = temp[idx] + "_tIOU_05"
                    results[-1][phase][name].append(item)
                    
                for idx, item in enumerate(R_tIOU_07_list):
                    name = temp[idx] + "_tIOU_07"
                    results[-1][phase][name].append(item)
                    
                print_results_test(results)
                
        df_row = [epoch+1, 
                np.mean(results[epoch]['test']['st_loss']), np.mean(results[epoch]['test']['ed_loss']), 
                np.mean(results[epoch]['test']['R1_tIOU_03']), np.mean(results[epoch]['test']['R5_tIOU_03']), np.mean(results[epoch]['test']['R10_tIOU_03']), np.mean(results[epoch]['test']['R50_tIOU_03']),
                np.mean(results[epoch]['test']['R1_tIOU_05']), np.mean(results[epoch]['test']['R5_tIOU_05']), np.mean(results[epoch]['test']['R10_tIOU_05']), np.mean(results[epoch]['test']['R50_tIOU_05']),
                np.mean(results[epoch]['test']['R1_tIOU_07']), np.mean(results[epoch]['test']['R5_tIOU_07']), np.mean(results[epoch]['test']['R10_tIOU_07']), np.mean(results[epoch]['test']['R50_tIOU_07']),
                #
                results[epoch]['test']['time'][-1]
                ]
        
        df.loc[len(df)] = round_converter(df_row, 4)

    print()
    save_df(df, "results_test.csv")
    
    epoch = np.argmin([results[i]["test"]["st_loss"][-1] for i in range(len(results))])

def train(net, dataloaders, model_name, optimizer, criterion, phases=["train", "valid", "test"], max_epochs=1000, classlabels=None):

    assert "train" in phases

    st_criterion = nn.CrossEntropyLoss()
    ed_criterion = nn.CrossEntropyLoss()
    fl_criterion = nn.CrossEntropyLoss()
    
    results = []
    
    df = pd.DataFrame(columns=['epoch', 
                               'train_st_loss', 'train_ed_loss', 'train_best_tIOU', 'train_mean_tIOU', 
                               'train_R@1_tIOU_03', 'train_R@5_tIOU_03', 'train_R@10_tIOU_03', 'train_R@50_tIOU_03',
                               'train_R@1_tIOU_05', 'train_R@5_tIOU_05', 'train_R@10_tIOU_05', 'train_R@50_tIOU_05',
                               'train_R@1_tIOU_07', 'train_R@5_tIOU_07', 'train_R@10_tIOU_07', 'train_R@50_tIOU_07',
                               #
                               'valid_st_loss', 'valid_ed_loss', 'valid_best_tIOU', 'valid_mean_tIOU', 
                               'valid_R@1_tIOU_03', 'valid_R@5_tIOU_03', 'valid_R@10_tIOU_03', 'valid_R@50_tIOU_03',
                               'valid_R@1_tIOU_05', 'valid_R@5_tIOU_05', 'valid_R@10_tIOU_05', 'valid_R@50_tIOU_05',
                               'valid_R@1_tIOU_07', 'valid_R@5_tIOU_07', 'valid_R@10_tIOU_07', 'valid_R@50_tIOU_07',
                               #
                               'test_st_loss', 'test_ed_loss', 'test_best_tIOU', 'test_mean_tIOU', 
                               'test_R@1_tIOU_03', 'test_R@5_tIOU_03', 'test_R@10_tIOU_03', 'test_R@50_tIOU_03',
                               'test_R@1_tIOU_05', 'test_R@5_tIOU_05', 'test_R@10_tIOU_05', 'test_R@50_tIOU_05',
                               'test_R@1_tIOU_07', 'test_R@5_tIOU_07', 'test_R@10_tIOU_07', 'test_R@50_tIOU_07',
                               #
                               'time'])
    
    for epoch in range(max_epochs):

        results.append(dict())
        R_accumulated = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Each epoch has a training and validation phase
        since = time.time()
        
        for phase in phases:

            # since = time.time()
            if phase == 'train':
                net.train(True)
            else:
                net.eval()

            fl_running_loss, st_running_loss, ed_running_loss = 0., 0., 0.
            st_num_correct, ed_num_correct = 0., 0.
            st_total_samples, ed_total_samples = 0., 0.

            o = np.zeros((0,))
            l = np.zeros((0,))

            o_ed = np.zeros((0, ))
            l_ed = np.zeros((0, ))

            #NOTE:  _tIOU_03, _tIOU_05, _tIOU_07
            results[-1][phase] = dict(st_loss=[], ed_loss=[], st_acc=[], ed_acc=[], tIOU=[], 
                                      R1_tIOU_03=[], R5_tIOU_03=[], R10_tIOU_03=[], R50_tIOU_03=[], 
                                      R1_tIOU_05=[], R5_tIOU_05=[], R10_tIOU_05=[], R50_tIOU_05=[], 
                                      R1_tIOU_07=[], R5_tIOU_07=[], R10_tIOU_07=[], R50_tIOU_07=[], 
                                      time=[])

            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):
                fl_running_loss, st_running_loss, ed_running_loss = 0., 0., 0.
                R_accumulated = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                inputs = data[0].view(-1, 1, 900, 205)
                targets = data[1]['target'] # tensor([4, 2, 3, 3, 0, ...]) 길이 64. 정답 label임.
                targets_onehot = F.one_hot(targets, num_classes=5)
                dur = data[1]['dur']
                labels = torch.stack([data[1]['st'][0],data[1]['st'][1]], dim=1) # torch.Size([64, 2]) 정답 시간 [[st, ed], [..]..]
                
                fl_labels = data[1]['fl_labels'].cuda()
                st_labels = data[1]['st_labels'].cuda()
                ed_labels = data[1]['ed_labels'].cuda()
                gap = data[1]['gap'].cuda()
                
                if phase != 'train':
                    with torch.no_grad():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                fl_outputs, st_outputs, ed_outputs = net(inputs, targets_onehot, dur) # output.shape = torch.Size([64, 900, 1])

                #NOTE: error

                fl_loss = fl_criterion(fl_outputs, fl_labels) 
                st_loss = st_criterion(st_outputs, st_labels)
                ed_loss = ed_criterion(ed_outputs, ed_labels)

                if phase == "train":
                    fl_loss.backward(retain_graph=True)
                    st_loss.backward(retain_graph=True)
                    ed_loss.backward(retain_graph=True)
                    optimizer.step()
                
                #NOTE: 
                # statistics
                fl_running_loss += (fl_loss.item() / (sum(gap)/30))
                st_running_loss += st_loss.item()
                ed_running_loss += ed_loss.item()
                
                # Accuracy: st
                st_num_correct += st_outputs[:st_labels.size(0)].max(dim=1)[1].eq(st_labels.max(1)[1]).sum().item()
                st_total_samples += len(st_outputs)
                
                o = np.concatenate((o, st_outputs[:st_labels.size(0)].max(1)[1].cpu().data.numpy()))
                l = np.concatenate((l, st_labels.max(1)[1].cpu().data.numpy()))

                # Accuracy: ed
                ed_num_correct += ed_outputs[:ed_labels.size(0)].max(dim=1)[1].eq(ed_labels.max(1)[1]).sum().item()
                ed_total_samples += len(ed_outputs)
                o_ed = np.concatenate((o_ed, ed_outputs[:ed_labels.size(0)].max(1)[1].cpu().data.numpy()))
                l_ed = np.concatenate((l_ed, ed_labels.max(1)[1].cpu().data.numpy()))
                
                st_list_pr = st_outputs.max(1)[1].cpu().data.numpy()
                st_list_gt = st_labels.max(1)[1].cpu().data.numpy()

                ed_list_pr = ed_outputs.max(1)[1].cpu().data.numpy()
                ed_list_gt = ed_labels.max(1)[1].cpu().data.numpy()
                
                # NOTE: Calculate tIOU
                tIOU = tIOU_calculator(st_list_pr, st_list_gt, ed_list_pr, ed_list_gt, dur)

                R_list = rank_tIOU(st_outputs, ed_outputs, st_labels, ed_labels, dur)
                B = data[1]['target'].shape
                # for R_l, R_a in zip(R_list, R_accumulated):
                #     R_a += R_l.sum().item()

                for idx, R_l in enumerate(R_list):
                    R_accumulated[idx] += R_l.sum().item() / B[0]
               
                results[-1][phase]["st_loss"].append(st_running_loss / (idx + 1))
                results[-1][phase]["st_acc"].append(st_num_correct / st_total_samples)
                
                results[-1][phase]["ed_loss"].append(ed_running_loss / (idx + 1))
                results[-1][phase]["ed_acc"].append(ed_num_correct / ed_total_samples)
                
                results[-1][phase]["time"].append(time.time() - since)
                results[-1][phase]["tIOU"].append(tIOU)
                
                # results[-1][phase]["R1"].append(R1)
                # results[-1][phase]["R10"].append(R10)
                
                R1_07, R5_07, R10_07, R50_07, R1_05, R5_05, R10_05, R50_05, R1_03, R5_03, R10_03, R50_03 = R_accumulated
                R_tIOU_03_list = [R1_03, R5_03, R10_03, R50_03]
                R_tIOU_05_list = [R1_05, R5_05, R10_05, R50_05]
                R_tIOU_07_list = [R1_07, R5_07, R10_07, R50_07]
                
                temp = ["R1", "R5", "R10", "R50"]
                for idx, item in enumerate(R_tIOU_03_list):
                    name = temp[idx] + "_tIOU_03"
                    results[-1][phase][name].append(item)
                
                for idx, item in enumerate(R_tIOU_05_list):
                    name = temp[idx] + "_tIOU_05"
                    results[-1][phase][name].append(item)
                    
                for idx, item in enumerate(R_tIOU_07_list):
                    name = temp[idx] + "_tIOU_07"
                    results[-1][phase][name].append(item)
                

                best_valid_loss, best_test_loss = print_results(results)
            
            #if best_valid_loss and epoch>30:
            #    matrix = confusion_matrix(l,o)

        #print(matrix)
        print("epoch: ", epoch)
        
        df_row = [epoch+1, 
                results[epoch]['train']['st_loss'][-1], results[epoch]['train']['ed_loss'][-1], results[epoch]['train']['tIOU'][-1], np.mean(results[epoch]['train']['tIOU']), 
                np.mean(results[epoch]['train']['R1_tIOU_03']), np.mean(results[epoch]['train']['R5_tIOU_03']), np.mean(results[epoch]['train']['R10_tIOU_03']), np.mean(results[epoch]['train']['R50_tIOU_03']),
                np.mean(results[epoch]['train']['R1_tIOU_05']), np.mean(results[epoch]['train']['R5_tIOU_05']), np.mean(results[epoch]['train']['R10_tIOU_05']), np.mean(results[epoch]['train']['R50_tIOU_05']),
                np.mean(results[epoch]['train']['R1_tIOU_07']), np.mean(results[epoch]['train']['R5_tIOU_07']), np.mean(results[epoch]['train']['R10_tIOU_07']), np.mean(results[epoch]['train']['R50_tIOU_07']),
                #
                results[epoch]['valid']['st_loss'][-1], results[epoch]['valid']['ed_loss'][-1], results[epoch]['valid']['tIOU'][-1], np.mean(results[epoch]['valid']['tIOU']), 
                np.mean(results[epoch]['valid']['R1_tIOU_03']), np.mean(results[epoch]['valid']['R5_tIOU_03']), np.mean(results[epoch]['valid']['R10_tIOU_03']), np.mean(results[epoch]['valid']['R50_tIOU_03']),
                np.mean(results[epoch]['valid']['R1_tIOU_05']), np.mean(results[epoch]['valid']['R5_tIOU_05']), np.mean(results[epoch]['valid']['R10_tIOU_05']), np.mean(results[epoch]['valid']['R50_tIOU_05']),
                np.mean(results[epoch]['valid']['R1_tIOU_07']), np.mean(results[epoch]['valid']['R5_tIOU_07']), np.mean(results[epoch]['valid']['R10_tIOU_07']), np.mean(results[epoch]['valid']['R50_tIOU_07']),
                #
                results[epoch]['test']['st_loss'][-1], results[epoch]['test']['ed_loss'][-1], results[epoch]['test']['tIOU'][-1], np.mean(results[epoch]['test']['tIOU']), 
                np.mean(results[epoch]['test']['R1_tIOU_03']), np.mean(results[epoch]['test']['R5_tIOU_03']), np.mean(results[epoch]['test']['R10_tIOU_03']), np.mean(results[epoch]['test']['R50_tIOU_03']),
                np.mean(results[epoch]['test']['R1_tIOU_05']), np.mean(results[epoch]['test']['R5_tIOU_05']), np.mean(results[epoch]['test']['R10_tIOU_05']), np.mean(results[epoch]['test']['R50_tIOU_05']),
                np.mean(results[epoch]['test']['R1_tIOU_07']), np.mean(results[epoch]['test']['R5_tIOU_07']), np.mean(results[epoch]['test']['R10_tIOU_07']), np.mean(results[epoch]['test']['R50_tIOU_07']),
                #
                results[epoch]['test']['time'][-1]
                ]
        
        df.loc[len(df)] = round_converter(df_row, 4)
        
        save_df(df, "results_train.csv")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, best_valid_loss, best_test_loss, model_name)

    print()
    epoch = np.argmin([results[i]["valid"]["loss"][-1] for i in range(len(results))])

    print(epoch+1,
          results[epoch]['train']['st_loss'][-1],
          results[epoch]['train']['ed_loss'][-1],
          results[epoch]['valid']['st_loss'][-1],
          results[epoch]['valid']['ed_loss'][-1],
          results[epoch]['valid']['st_acc'][-1],
          results[epoch]['valid']['ed_acc'][-1],
          results[epoch]['test']['st_loss'][-1] if 'test' in phases else "",
          results[epoch]['test']['ed_loss'][-1] if 'test' in phases else "",
          results[epoch]['test']['st_acc'][-1] if 'test' in phases else "" ,
          results[epoch]['test']['ed_acc'][-1] if 'test' in phases else "" )
