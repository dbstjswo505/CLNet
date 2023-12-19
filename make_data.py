import json
import scipy.io as sio
import os
import pdb
import numpy as np
import random
from random import randint
from random import seed
seed(1)

pth = './IDRad/'
lb_list = os.listdir('./IDRad')
lb_list.sort()

def get_corpus(lb_list, target):
    ck = 'target'+ str(target)
    cat_data = np.zeros((30000,256))
    L = 0
    for i in range(len(lb_list)):
        name = lb_list[i][:7]
        if name == ck:
            data = sio.loadmat(pth+lb_list[i])['doppler_threshold']
            l = data.shape[0]
            cat_data[L:L+l,:] = data
    
            L = L + l
    return cat_data[:L,:]

def overlap(data, template, start, ovl=10):
    L = data.shape[0]
    if start < 10:
        template[start:start+L,:] = data
        gs = start
        ge = gs + L
    else:
        for i in range(ovl):
            template[start-ovl+i] = (1-i/ovl)*template[start-ovl+i] + (i/ovl)*data[i]
        template[start:start+L-ovl] = data[ovl:]
        gs = start-ovl
        ge = gs + L
    return template, gs, ge

def get_data(corpus, get_target, min_L=40, max_L=200):
    out_data = np.zeros((1000,256))
    mk = 0
    gst = list()
    for i in range(len(get_target)):
        target = get_target[i]
        corpus_radar = corpus[target]
        total_L = corpus_radar.shape[0]
        dur = randint(min_L,max_L)
        range_L = total_L - dur - 1
        start = randint(0,range_L)
        end = start + dur

        data_ = corpus[target][start:end]

        ovl = random.randint(10,20)
        out_data, gs, ge = overlap(data_, out_data, mk, ovl)

        gse = [gs,ge]
        gst.append(gse)

        mk = ge
    return out_data[:mk,:], gst



target1_corpus = get_corpus(lb_list,1)
target2_corpus = get_corpus(lb_list,2)
target3_corpus = get_corpus(lb_list,3)
target4_corpus = get_corpus(lb_list,4)
target5_corpus = get_corpus(lb_list,5)
target_corpus = [target1_corpus[:15000], target2_corpus[:15000], target3_corpus[:15000], target4_corpus[:15000], target5_corpus[:15000]]
test_target_corpus = [target1_corpus[15000:], target2_corpus[15000:], target3_corpus[15000:], target4_corpus[15000:], target5_corpus[15000:]]
target_list = [0,1,2,3,4]

train = 4000
val = 500
test = 500
data_id = 0

train_pth = './IDRad-TBA/train/'
train_list = list()
for i in range(train):
    num = randint(1,5)
    get_list = random.sample(target_list, k=num)
    get_target = [*set(get_list)]
    random.shuffle(get_target)
    untrimmed_radar, ann = get_data(target_corpus, get_target)
    np.save(train_pth + str(data_id)+'.npy', untrimmed_radar)
    for j in range(len(ann)):
        sample_dict = dict()
        sample_dict['radar_id'] = data_id
        sample_dict['st'] = ann[j]
        sample_dict['target'] = get_target[j]
        sample_dict['dur'] = untrimmed_radar.shape[0]
        train_list.append(sample_dict)
    
    # update
    data_id = data_id + 1
with open("./IDRad-TBA/train.json", 'w') as train_f:
    json.dump(train_list, train_f)

# val
train_pth = './IDRad-TBA/val/'
train_list = list()
for i in range(val):
    num = randint(1,5)
    get_list = random.sample(target_list, k=num)
    get_target = [*set(get_list)]
    random.shuffle(get_target)
    untrimmed_radar, ann = get_data(target_corpus, get_target)
    np.save(train_pth + str(data_id)+'.npy', untrimmed_radar)
    for j in range(len(ann)):
        sample_dict = dict()
        sample_dict['radar_id'] = data_id
        sample_dict['st'] = ann[j]
        sample_dict['target'] = get_target[j]
        sample_dict['dur'] = untrimmed_radar.shape[0]
        train_list.append(sample_dict)
    
    # update
    data_id = data_id + 1
with open("./IDRad-TBA/val.json", 'w') as train_f:
    json.dump(train_list, train_f)

# test
train_pth = './IDRad-TBA/test/'
train_list = list()
for i in range(val):
    num = randint(1,5)
    get_list = random.sample(target_list, k=num)
    get_target = [*set(get_list)]
    random.shuffle(get_target)
    untrimmed_radar, ann = get_data(test_target_corpus, get_target)
    np.save(train_pth + str(data_id)+'.npy', untrimmed_radar)
    for j in range(len(ann)):
        sample_dict = dict()
        sample_dict['radar_id'] = data_id
        sample_dict['st'] = ann[j]
        sample_dict['target'] = get_target[j]
        sample_dict['dur'] = untrimmed_radar.shape[0]
        train_list.append(sample_dict)
    
    # update
    data_id = data_id + 1
with open("./IDRad-TBA/test.json", 'w') as train_f:
    json.dump(train_list, train_f)


#target 
pdb.set_trace()


