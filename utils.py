import numpy as np
import os
import torch
import matplotlib.pylab as plt
import shutil
from easydict import EasyDict
import json


class ansi:
    BLACK = '\033[30m'
    GRAY = '\033[37m'
    DARKGRAY = '\033[90m'
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def load_config(config_json_file) -> EasyDict:
    with open(config_json_file, "r", encoding='utf-8') as reader:
        config = json.loads(reader.read())
    cfg = EasyDict(config)

    return cfg

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_checkpoint(state, is_best_valid, is_best_test, model_name):

    torch.save(state, os.path.join('params/', model_name + ".pt"))

    if is_best_valid:
        shutil.copyfile(os.path.join('params/', model_name + ".pt"), os.path.join('params/', model_name + "_bvalid.pt"))

    if is_best_test:
        shutil.copyfile(os.path.join('params/', model_name + ".pt"), os.path.join('params/', model_name + "_btest.pt"))



def load_checkpoint(net, optimizer, filename, is_cuda=True, remove_module=False, add_module=False):

    if os.path.isfile(filename):
        checkpoint = torch.load(filename) if is_cuda else torch.load(filename, map_location=lambda storage, loc: storage)
        model_state = net.state_dict()

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if remove_module:
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        if add_module:
            state_dict = {'module.' + k: v for k, v in state_dict.items() }

        for k, v in state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                # print("[INFO] Loading param %s with size %s into model."%(k, ','.join(map(str, model_state[k].size()))))
                pass
            else:
                # print("Size in model is ", v.size(), filename)
                print("[WARNING] Could not load params %s in model." % k)

        pretrained_state = {k: v for k, v in state_dict.items() if
                            k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        net.load_state_dict(model_state)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("[WARNING] Could not find params file %s." % filename)


def print_results(results):

    has_train = 'train' in results[-1] and len(results[-1]['train']) > 0
    has_valid = 'valid' in results[-1] and len(results[-1]['valid']) > 0
    has_test = 'test' in results[-1] and len(results[-1]['test']) > 0

    # labels = [l for l in results[-1][list(results[-1].keys())[0]].keys() if l not in ["st_loss", "ed_loss", "st_acc", "ed_acc", "time"]]
    # labels = [l for l in results[-1][list(results[-1].keys())[0]].keys() if l in ["st_loss", "ed_loss", "st_acc", "ed_acc", "time"]]

    # labels_header = "" # labels_header = " st_loss | ed_loss | time | st_acc | ed_acc |"
    # second_header = "" # second_header = "---------|---------|------|--------|--------|"
    # format_header = "" # format_header = "{:>4.2f} {:>4.2f}|{:>4.2f} {:>4.2f}|{:>4.2f} {:>4.2f}|{:>4.2f} {:>4.2f}|{:>4.2f} {:>4.2f}|"

    # for label in labels:
    #     length = max(9, len(label) + 2)
    #     spaces = length - len(label)
    #     label_header = " " * (spaces // 2) + label + " " * (spaces - spaces // 2) + "|"
    #     labels_header += label_header
    #     second_header += "-" * (len(label_header) - 1) + "|"
    #     format_header += "{:>" + str(length//2) + ".2f} {:>" + str(length//2 - (1 - length % 2)) + ".2f}|"

    if len(results) == 1 and len(results[-1]['train']['st_loss']) == 1:

        # v1.0 original
        # valid_header = " Valid st_loss | Valid ed_loss | Valid err |"
        # valid_line =   "---------------|---------------|-----------|"
        # test_header = " Test st_loss | Test st_loss | Test err |"
        # test_line =   "--------------|--------------|----------|"

        # print(" Epoch  | Batch | Train st_loss | Train st_loss | Train err |%s%s  Dur       %s\n"
        #       "--------|-------|---------------|---------------|-----------|%s%s--------    %s" % (valid_header, test_header, labels_header[:-1], valid_line, test_line, second_header[:-1]))

        # v1.1 modified
        # valid_header = " Valid st_loss | Valid ed_loss | Valid st_acc | Valid ed_acc |"
        # valid_line =   "---------------|---------------|--------------|--------------|"
        # test_header = " Test st_loss | Test ed_loss | Test st_acc | Test ed_acc |"
        # test_line =   "--------------|--------------|-------------|-------------|"
        
        # print("Epoch   | Batch | Train st_loss | Train ed_loss | Train st_acc | Train ed_acc |%s%s  Dur       \n"
        #       "--------|-------|---------------|---------------|--------------|--------------|%s%s--------    " % (valid_header, test_header, valid_line, test_line))

        #v1.2 modified
        # valid_header = " Valid st_loss | Valid ed_loss | Valid best_tIOU | Valid mean_tIOU |"
        # valid_line =   "---------------|---------------|-----------------|-----------------|"
        # test_header = " Test st_loss | Test ed_loss | Test best_tIOU | Test mean_tIOU |"
        # test_line =   "--------------|--------------|----------------|----------------|"
        
        # print("Epoch   | Batch | Train st_loss | Train ed_loss | Train best_tIOU | Train mean_tIOU |%s%s  Dur       \n"
        #       "--------|-------|---------------|---------------|-----------------|-----------------|%s%s--------    " % (valid_header, test_header, valid_line, test_line))
        
        #v1.3 modified
        valid_header = " Valid st_loss | Valid ed_loss | Valid m_tIOU | Valid R@1 | Valid R@10 |"
        valid_line =   "---------------|---------------|--------------|-----------|------------|"
        test_header = " Test m_tIOU | Test R@1 | Test R@10 |"
        test_line =   "-------------|----------|-----------|"
        
        print("Epoch   | Batch | Train st_loss | Train ed_loss | Train m_tIOU | Train R@1 | Train R@10 |%s%s  Dur       \n"
              "--------|-------|---------------|---------------|--------------|-----------|------------|%s%s--------    " % (valid_header, test_header, valid_line, test_line))
        
    # Results of "start time" classification
    st_train_loss_results = [results[i]["train"]["st_loss"][-1] for i in range(len(results) - 1)] if has_train else []
    st_best_train_loss = has_train and (results[-1]["train"]["st_loss"][-1] <= (np.min(st_train_loss_results) if len(st_train_loss_results) > 0 else np.inf))
    # st_train_acc_results = [results[i]["train"]["st_acc"][-1] for i in range(len(results))] if has_train else []
    # st_best_train_acc = has_train and (results[-1]["train"]["st_acc"][-1] <= np.min(st_train_acc_results))

    st_valid_loss_results = [results[i]["valid"]["st_loss"][-1] for i in range(len(results) - 1)] if has_valid else []
    st_best_valid_loss = has_valid and (results[-1]["valid"]["st_loss"][-1] <= (np.min(st_valid_loss_results) if len(st_valid_loss_results) > 0 else np.inf))
    # st_valid_acc_results = [results[i]["valid"]["st_acc"][-1] for i in range(len(results))] if has_valid else[]
    # st_best_valid_acc = has_valid and (results[-1]["valid"]["st_acc"][-1] <= np.min(st_valid_acc_results))

    st_test_loss_results = [results[i]["test"]["st_loss"][-1] for i in range(len(results) - 1)] if has_test else []
    st_best_test_loss = has_test and (results[-1]["test"]["st_loss"][-1] <= (np.min(st_test_loss_results) if len(st_test_loss_results) > 0 else np.inf))
    # st_test_acc_results = [results[i]["test"]["st_acc"][-1] for i in range(len(results))] if has_test else []
    # st_best_test_acc = has_test and (results[-1]["test"]["st_acc"][-1] <= np.min(st_test_acc_results))


    # Results of "end time" classification
    ed_train_loss_results = [results[i]["train"]["ed_loss"][-1] for i in range(len(results) - 1)] if has_train else []
    ed_best_train_loss = has_train and (results[-1]["train"]["ed_loss"][-1] <= (np.min(ed_train_loss_results) if len(ed_train_loss_results) > 0 else np.inf))
    # ed_train_acc_results = [results[i]["train"]["ed_acc"][-1] for i in range(len(results))] if has_train else []
    # ed_best_train_acc = has_train and (results[-1]["train"]["ed_acc"][-1] <= np.min(ed_train_acc_results))

    ed_valid_loss_results = [results[i]["valid"]["ed_loss"][-1] for i in range(len(results) - 1)] if has_valid else []
    ed_best_valid_loss = has_valid and (results[-1]["valid"]["ed_loss"][-1] <= (np.min(ed_valid_loss_results) if len(ed_valid_loss_results) > 0 else np.inf))
    # ed_valid_acc_results = [results[i]["valid"]["ed_acc"][-1] for i in range(len(results))] if has_valid else[]
    # ed_best_valid_acc = has_valid and (results[-1]["valid"]["ed_acc"][-1] <= np.min(ed_valid_acc_results))

    ed_test_loss_results = [results[i]["test"]["ed_loss"][-1] for i in range(len(results) - 1)] if has_test else []
    ed_best_test_loss = has_test and (results[-1]["test"]["ed_loss"][-1] <= (np.min(ed_test_loss_results) if len(ed_test_loss_results) > 0 else np.inf))
    # ed_test_acc_results = [results[i]["test"]["ed_acc"][-1] for i in range(len(results))] if has_test else []
    # ed_best_test_acc = has_test and (results[-1]["test"]["ed_acc"][-1] <= np.min(ed_test_acc_results))

    # Results of "tIOU"
    train_tIOU_results = [results[i]["train"]["tIOU"][-1] for i in range(len(results))] if has_train else []
    best_train_tIOU = has_train and (results[-1]["train"]["tIOU"][-1] <= np.max(train_tIOU_results))
    mean_train_tIOU = np.mean(results[-1]["train"]["tIOU"]) if has_train else 0.0 

    valid_tIOU_results = [results[i]["valid"]["tIOU"][-1] for i in range(len(results))] if has_valid else []
    best_valid_tIOU = has_valid and (results[-1]["valid"]["tIOU"][-1] <= np.max(valid_tIOU_results))
    mean_valid_tIOU = np.mean(results[-1]["valid"]["tIOU"]) if has_valid else 0.0
    
    test_tIOU_results = [results[i]["test"]["tIOU"][-1] for i in range(len(results))] if has_test else []
    best_test_tIOU = has_test and (results[-1]["test"]["tIOU"][-1] <= np.max(test_tIOU_results))
    mean_test_tIOU = np.mean(results[-1]["test"]["tIOU"]) if has_test else 0.0

    # Results of "R@1_tIOU_03"
    train_R1_results = [results[i]["train"]["R1_tIOU_03"][-1] for i in range(len(results))] if has_train else []
    mean_train_R1 = np.mean(results[-1]["train"]["R1_tIOU_03"]) if has_train else 0.0
    
    valid_R1_results = [results[i]["valid"]["R1_tIOU_03"][-1] for i in range(len(results))] if has_valid else []
    mean_valid_R1 = np.mean(results[-1]["valid"]["R1_tIOU_03"]) if has_valid else 0.0
    
    test_R1_results = [results[i]["test"]["R1_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R1 = np.mean(results[-1]["test"]["R1_tIOU_03"]) if has_test else 0.0
    
    # Results of "R@10_tIOU_03"
    train_R10_results = [results[i]["train"]["R10_tIOU_03"][-1] for i in range(len(results))] if has_train else []
    mean_train_R10 = np.mean(results[-1]["train"]["R10_tIOU_03"]) if has_train else 0.0
    
    valid_R10_results = [results[i]["valid"]["R10_tIOU_03"][-1] for i in range(len(results))] if has_valid else []
    mean_valid_R10 = np.mean(results[-1]["valid"]["R10_tIOU_03"]) if has_valid else 0.0
    
    test_R10_results = [results[i]["test"]["R10_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R10 = np.mean(results[-1]["test"]["R10_tIOU_03"]) if has_test else 0.0


    # class_accuracies = []
    # for label in labels:

    #     if not has_test:
    #         class_accuracies.append(results[-1]['train'][label][-1] if has_train else 1.0)

    #     class_accuracies.append(results[-1]['valid'][label][-1] if has_valid else 1.0)

    #     if has_test:
    #         class_accuracies.append(results[-1]['test'][label][-1] if has_test else 1.0)

    # v1.0
    # valid_format =  " {}{:>10.5f}{} | {}{:>10.5f}{} | {}{:>9.4f}{} | {}{:>9.4f}{} |"
    # test_format =   " {}{:>9.5f}{} | {}{:>9.5f}{} | {}{:>8.4f}{} | {}{:>8.4f}{} |"

    # v1.1
    # valid_format =  " {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>12f}{} | {}{:>12f}{} |"
    # test_format =   " {}{:>12f}{} | {}{:>12f}{} | {}{:>11f}{} | {}{:>11f}{} |"

    # v1.2
    # valid_format =  " {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} |"
    # test_format =   " {}{:>12f}{} | {}{:>12f}{} | {}{:>14f}{} | {}{:>14f}{} | {}{:>14f}{} | {}{:>14f}{} |"
    
    # v1.3
    valid_format =  " {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>12f}{} | {}{:>9f}{} | {}{:>10f}{} |"
    test_format =   " {}{:>11f}{} | {}{:>8f}{} | {}{:>9f}{} |"

    total_time = np.sum([results[-1][phase]["time"][-1] for phase in ["train", "valid", "test"] if phase in results[-1]])
    total_batch = np.sum([len(results[-1][phase]["st_loss"]) for phase in ["train", "valid", "test"] if phase in results[-1]])

    # v1.0
    # print((" {:>6} | {:>5} | {}{:>10.5f}{} | {}{:>9.4f}{} |" + valid_format + test_format + " {:>6.1f}s    " + format_header[:-1]).format(
    
    # v1.1
    # print((" {:>6} | {:>5} | {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>12f}{} | {}{:>12f}{} |" + valid_format + test_format + " {:>6.1f}    ").format(
        
    # v1.2
    # print((" {:>6} | {:>5} | {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} |" + valid_format + test_format + " {:>6.1f}    ").format(
        
    # v1.3
    print((" {:>6} | {:>5} | {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>12f}{} | {}{:>9f}{} | {}{:>10f}{} |" + valid_format + test_format + " {:>6.1f}    ").format(
        len(results), total_batch,
        
        # 1-1. Train: st / ed train_loss
        ansi.BLACK if st_best_train_loss else ansi.GRAY,
        results[-1]["train"]["st_loss"][-1] if has_train else -1.0,
        ansi.ENDC,
        ansi.BLACK if ed_best_train_loss else ansi.GRAY,
        results[-1]["train"]["ed_loss"][-1] if has_train else -1.0,
        ansi.ENDC,
        
        # 1-2. Train: st / ed train_acc
        # ansi.RED if st_best_train_acc else ansi.GRAY,
        # results[-1]["train"]["st_acc"][-1] if has_train else -1.0,
        # ansi.ENDC,
        # ansi.RED if ed_best_train_acc else ansi.GRAY,
        # results[-1]["train"]["ed_acc"][-1] if has_train else -1.0,
        # ansi.ENDC,
        
        # 1-3. Train: st / ed train_tIOU
        # ansi.RED if best_train_tIOU else ansi.GRAY,
        # results[-1]["train"]["tIOU"][-1] if has_train else -1.0,
        # ansi.ENDC,
        
        ansi.BLUE,
        mean_train_tIOU,
        ansi.ENDC,
        
        ansi.RED,
        mean_train_R1,
        ansi.ENDC,
        
        ansi.PURPLE,
        mean_train_R10,
        ansi.ENDC,
        
        # 2-1. Valid: st/ed valid_loss
        ansi.GREEN if st_best_valid_loss else ansi.GRAY,
        results[-1]["valid"]["st_loss"][-1] if has_valid else -1.0,
        ansi.ENDC,
        ansi.GREEN if ed_best_valid_loss else ansi.GRAY,
        results[-1]["valid"]["ed_loss"][-1] if has_valid else -1.0,
        ansi.ENDC,
        
        # 2-2. Valid: st/ed valid_acc
        # ansi.RED if st_best_valid_acc else ansi.GRAY,
        # results[-1]["valid"]["st_acc"][-1] if has_valid else -1.0,
        # ansi.ENDC,
        # ansi.RED if ed_best_valid_acc else ansi.GRAY,
        # results[-1]["valid"]["ed_acc"][-1] if has_valid else -1.0,
        # ansi.ENDC,
        
        # 2-3. Valid: st/ed valid_tIOU
        # ansi.RED if best_valid_tIOU else ansi.GRAY,
        # results[-1]["valid"]["tIOU"][-1] if has_valid else -1.0,
        # ansi.ENDC,
    
        ansi.BLUE,
        mean_valid_tIOU,
        ansi.ENDC,
    
        ansi.RED,
        mean_valid_R1,
        ansi.ENDC,
        
        ansi.PURPLE,
        mean_valid_R10,
        ansi.ENDC,
    
        # 3-1. Test: st/ed test_loss
        # ansi.GREEN if st_best_test_loss else ansi.GRAY,
        # results[-1]["test"]["st_loss"][-1] if has_test else -1.0,
        # ansi.ENDC,
        # ansi.GREEN if ed_best_test_loss else ansi.GRAY,
        # results[-1]["test"]["ed_loss"][-1] if has_test else -1.0,
        # ansi.ENDC,
        
        # 3-2. Test: st/ed test_acc
        # ansi.RED if st_best_test_acc else ansi.GRAY,
        # results[-1]["test"]["st_acc"][-1] if has_test else -1.0,
        # ansi.ENDC,
        # ansi.RED if ed_best_test_acc else ansi.GRAY,
        # results[-1]["test"]["ed_acc"][-1] if has_test else -1.0,
        # ansi.ENDC,
        
        # 3-3. Test: st/ed test_tIOU
        # ansi.RED if best_test_tIOU else ansi.GRAY,
        # results[-1]["test"]["tIOU"][-1] if has_test else -1.0,
        # ansi.ENDC,
        
        ansi.BLUE,
        mean_test_tIOU,
        ansi.ENDC,
        
        ansi.RED,
        mean_test_R1,
        ansi.ENDC,
        
        ansi.PURPLE,
        mean_test_R10,
        ansi.ENDC,
        
        round(total_time, 2),
        ), end='\r') # *class_accuracies


    return st_best_valid_loss, st_best_test_loss



def print_results_test(results):

    has_train = 'train' in results[-1] and len(results[-1]['train']) > 0
    has_valid = 'valid' in results[-1] and len(results[-1]['valid']) > 0
    has_test = 'test' in results[-1] and len(results[-1]['test']) > 0

    if len(results) == 1 and len(results[-1]['test']['st_loss']) == 1:

        #v1.3 modified
        test_header = " Test b_tIOU | Test m_tIOU | Test R@1 | Test R@5 | Test R@10 | Test R@50 |"
        test_line =   "-------------|-------------|----------|----------|-----------|-----------|"
                      
        print("Epoch   | Batch | Test st_loss | Test ed_loss |%s  Dur       \n"
              "--------|-------|--------------|--------------|%s--------    " % (test_header, test_line))

    # Results of "start time" classification
    st_test_loss_results = [results[i]["test"]["st_loss"][-1] for i in range(len(results) - 1)] if has_test else []
    st_best_test_loss = has_test and (results[-1]["test"]["st_loss"][-1] <= (np.min(st_test_loss_results) if len(st_test_loss_results) > 0 else np.inf))
    # st_test_acc_results = [results[i]["test"]["st_acc"][-1] for i in range(len(results))] if has_test else []
    # st_best_test_acc = has_test and (results[-1]["test"]["st_acc"][-1] <= np.min(st_test_acc_results))


    # Results of "end time" classification
    ed_test_loss_results = [results[i]["test"]["ed_loss"][-1] for i in range(len(results) - 1)] if has_test else []
    ed_best_test_loss = has_test and (results[-1]["test"]["ed_loss"][-1] <= (np.min(ed_test_loss_results) if len(ed_test_loss_results) > 0 else np.inf))
    # ed_test_acc_results = [results[i]["test"]["ed_acc"][-1] for i in range(len(results))] if has_test else []
    # ed_best_test_acc = has_test and (results[-1]["test"]["ed_acc"][-1] <= np.min(ed_test_acc_results))

    # Results of "tIOU"
    test_tIOU_results = [results[i]["test"]["tIOU"][-1] for i in range(len(results))] if has_test else []
    best_test_tIOU = has_test and (results[-1]["test"]["tIOU"][-1] <= np.max(test_tIOU_results))
    mean_test_tIOU = np.mean(results[-1]["test"]["tIOU"]) if has_test else 0.0

    # Results of "R@1_tIOU_03"
    test_R1_results = [results[i]["test"]["R1_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R1 = np.mean(results[-1]["test"]["R1_tIOU_03"]) if has_test else 0.0
    
    # Results of "R@5_tIOU_03"
    test_R5_results = [results[i]["test"]["R5_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R5 = np.mean(results[-1]["test"]["R5_tIOU_03"]) if has_test else 0.0
    
    # Results of "R@10_tIOU_03"
    test_R10_results = [results[i]["test"]["R10_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R10 = np.mean(results[-1]["test"]["R10_tIOU_03"]) if has_test else 0.0
    
    # Results of "R@50_tIOU_03"   
    test_R50_results = [results[i]["test"]["R50_tIOU_03"][-1] for i in range(len(results))] if has_test else []
    mean_test_R50 = np.mean(results[-1]["test"]["R50_tIOU_03"]) if has_test else 0.0

    # v1.3           st_loss 
    test_format =   "{}{:>13f}{} | {}{:>12f}{} | {}{:>11f}{} | {}{:>11f}{} | {}{:>8f}{} | {}{:>8f}{} | {}{:>9f}{} | {}{:>9f}{} |"

    total_time = np.sum([results[-1][phase]["time"][-1] for phase in ["train", "valid", "test"] if phase in results[-1]])
    total_batch = np.sum([len(results[-1][phase]["st_loss"]) for phase in ["train", "valid", "test"] if phase in results[-1]])

    # v1.0
    # print((" {:>6} | {:>5} | {}{:>10.5f}{} | {}{:>9.4f}{} |" + valid_format + test_format + " {:>6.1f}s    " + format_header[:-1]).format(
    
    # v1.1
    # print((" {:>6} | {:>5} | {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>12f}{} | {}{:>12f}{} |" + valid_format + test_format + " {:>6.1f}    ").format(
        
    # v1.2
    # print((" {:>6} | {:>5} | {}{:>13.5f}{} | {}{:>13.5f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} | {}{:>15f}{} |" + valid_format + test_format + " {:>6.1f}    ").format(
        
    # v1.3
    print((" {:>6} | {:>5} |" + test_format + " {:>6.1f}    ").format(
        len(results), total_batch,
        
        # 3-1. Test: st/ed test_loss
        # ansi.GREEN if st_best_test_loss else ansi.GRAY,
        # results[-1]["test"]["st_loss"][-1] if has_valid else -1.0,
        # ansi.ENDC,
        # ansi.GREEN if ed_best_test_loss else ansi.GRAY,
        # results[-1]["test"]["ed_loss"][-1] if has_valid else -1.0,
        # ansi.ENDC, 
        
        ansi.GREEN,
        results[-1]['test']['st_loss'][-1],
        ansi.ENDC,
        
        ansi.GREEN,
        results[-1]['test']['ed_loss'][-1],
        ansi.ENDC,
        
        ansi.BLUE,
        np.max(test_tIOU_results),
        ansi.ENDC,
        
        ansi.BLUE,
        mean_test_tIOU,
        ansi.ENDC,
        
        ansi.RED,
        mean_test_R1,
        ansi.ENDC,
        
        ansi.RED,
        mean_test_R5,
        ansi.ENDC,
        
        ansi.RED,
        mean_test_R10,
        ansi.ENDC,
        
        ansi.RED,
        mean_test_R50,
        ansi.ENDC,
        
        round(total_time, 2),
        ), end='\r') # *class_accuracies

