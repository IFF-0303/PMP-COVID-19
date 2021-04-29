import pickle
import os
import math
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from model import MedicNet, MultiFocalLoss
from copy import deepcopy
from torchnet import meter
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from warmup_scheduler import GradualWarmupScheduler
from utils import dataset, split_data, register, load_data, data_normalize, split_by_types, load_data1_split_idx, \
                  CTs_augmentation_batch, seed_torch, RepeatedKFold_class_wise_split_by_age

def load_checkpoint(path):
    model_file = sorted(os.listdir(path))[-1]
    config_file = sorted(os.listdir(path))[0]
    model_file = os.path.join(path, model_file)
    config_file = os.path.join(path, config_file)
    print('Load Model and Config file...')
    print(model_file)
    print(config_file)
    with open(config_file, 'rb') as fp:
        config = pickle.load(fp)
    model = MedicNet(config)
    model.load_state_dict(torch.load(model_file))
    return model.cuda(), config

def val(model, val_data, config, epoch, best_acc, best_sen, best_val_epoch, best_model_state_dict, ret_dict, data_name, num_fold):
    with torch.no_grad():
        model.eval()
        score_list=[]
        comatrix_list=[]
        i=0

        for val_batch in tqdm(val_data):
            val_batch['ct_mask'] = val_batch['ct_mask'].cuda()
            val_batch['CTs'] = val_batch['CTs'].cuda()          
            score, _ = model(val_batch)
            confusion_matrix = meter.ConfusionMeter(config.num_classes)
            confusion_matrix.add(score.detach().cpu(), val_batch['labels'].cpu())
            comatrix_list.append(confusion_matrix.value())

            if i ==0:
                scores=score.detach().cpu()
                labels=val_batch['labels'].cpu()
            else:
                scores=torch.cat((scores,score.detach().cpu()),axis=0)
                labels=torch.cat((labels,val_batch['labels'].cpu()),axis=0)

            i+=1

        model.train()
        cm_all = np.array(comatrix_list).sum(axis=0)
        print('Epoch ', epoch+1)
        print('%s confusion matrix: ' % (data_name))
        print(cm_all)

        acc_epoch = (float(cm_all[0, 0])+float(cm_all[1, 1]))/cm_all.sum()
        spec_epoch = float(cm_all[0, 0]) / (cm_all[0, 0] + cm_all[0, 1]) # TN：val_cm[0, 0], FP：val_cm[0, 1],
        sen_epoch = float(cm_all[1, 1]) / (cm_all[1, 0] + cm_all[1, 1])  # FN：val_cm[1, 0], TP：val_cm[1, 1]
        precision_epoch = float(cm_all[1, 1]) / (cm_all[0, 1] + cm_all[1, 1]) 
      
        print('%s accuracy: %.3f' % (data_name, acc_epoch))
        print('%s sensitivity: %.3f' % (data_name, sen_epoch))
        print('%s specificity: %.3f' % (data_name, spec_epoch))
        print('%s precision: %.3f' % (data_name, precision_epoch))
        
        if acc_epoch>=best_acc and data_name=='val':
            best_model_state_dict = deepcopy(model.state_dict())
            best_acc=acc_epoch
            best_sen=sen_epoch
            best_val_epoch=epoch
            ret_dict['sens'] = sen_epoch
            ret_dict['specs'] = spec_epoch
            ret_dict['accs'] = best_acc
            ret_dict['precisions'] = precision_epoch            
            ret_dict['preds'] = scores.detach().cpu().numpy()
            ret_dict['labels'] = labels.cpu().numpy()
            
            # save model
            root = './checkpoints/%s/fold%d' % (check_dir, num_fold)
            if not os.path.exists(root):
                os.mkdir(root)
                
            torch.save(best_model_state_dict, '%s/acc%.3f-sen%.3f@epoch%d.pth' % \
                (root, best_acc,best_sen,best_val_epoch+1))
            
            # save config
            with open('checkpoints/%s/config.pickle' % check_dir, 'wb') as fp:
                pickle.dump(config, fp)

            # save result
            res_path = './res/%s/fold%d' % (check_dir, num_fold)
            if not os.path.exists(res_path):
                os.mkdir(res_path) 
            saved_val = register(dump_file='./res/%s/fold%d/val_dataset1_acc%.3f-sen%.3f.pickle' % (check_dir, num_fold, best_acc, best_sen))
            saved_val.regis(ret_dict)
            saved_val.dump()

    return best_model_state_dict, best_acc, best_sen, best_val_epoch, ret_dict

def train(model, train_data, val_data, config, num_fold):
    optimizer = optim.Adam(model.parameters(),lr=config.lr, weight_decay = 1e-4)
    scheduler_StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
    
    if config.enable_focal_loss == True:
        criterion = MultiFocalLoss(config.num_classes, alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion_CE = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(dataset(train_data, config), batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset(val_data, config), batch_size=config.batch_size, shuffle=False)
    
    best_model_state_dict = model.state_dict()
    best_acc = -1
    best_sen = -1
    ret_dict = {'accs': -1, 'sens':-1, 'specs':-1, 'precisions':-1, 'preds':None, 'labels':None}
    best_val_epoch = 0
    iters = len(train_dataloader)

    for epoch in range(config.max_epoch):
        for data_src in tqdm(train_dataloader):
            if config.CTs_augmentation==True:
                data_src = CTs_augmentation_batch(config, data_src)

            class_src = data_src['labels']
            data_src['ct_mask'] = data_src['ct_mask'].cuda()
            optimizer.zero_grad()
            src_class_output, features = model(data_src)

            loss = criterion_CE(src_class_output, class_src)
            loss.backward()
            optimizer.step()

        print('Val on source data ...')
        best_model_state_dict, best_acc, best_sen, best_val_epoch, ret_dict = val(model, val_dataloader, config, epoch, best_acc, best_sen, best_val_epoch, best_model_state_dict, ret_dict, 'val', num_fold)
        print('Accuracy %.3f sensitivity %.3f @ epoch %d' % (best_acc,best_sen,best_val_epoch+1))
        scheduler_StepLR.step()
        print('Current lr: ', optimizer.param_groups[0]['lr'])

    return model, ret_dict

if __name__ == "__main__":
    seed_torch(seed=7777)
    remove_items = []
    config = edict()
    # path setting
    config.data_root = ''
    config.pretrained_model_path = ''
    
    # training setting
    config.lr = 0.05 # 1e-3
    config.fc_bias = False
    config.clinica_feat_dim = 61
    config.CT_feat_dim = 128
    config.lstm_indim = config.clinica_feat_dim+config.CT_feat_dim
    config.hidden_dim = config.lstm_indim*2
    config.num_classes = 2
    config.seq_len = 7
    config.seq_processor = 'lstm' # gru / lstm
    config.mlp_indim = 79 - len(remove_items)
    config.few_shot_number = 10
    config.batch_size = 32
    config.max_epoch = 100
    config.repeat = 1
    config.densenet_drop_rate = 0.5   
    config.encoder_3d = 'resnet' # baseline / densenet / resnet
    config.sdata_pool = 'avg' # avg / max
    config.ddata_pool = 'avg' # avg / max
    config.init_tau = 5
    config.clinical_att = False
    config.clinical_backbone = 'mlp+res' # resnet / mlp+res
    config.lstm_all_output = True
    config.lstm_att = False
    config.enable_focal_loss = False
    config.clinical_augmentation = True

    # dataset setting
    config.dataset_split_rate = 0.8
    config.ddata1 = config.data_root + 'handcraft/ddata1.pickle'
    config.ddata1_woPCA = config.data_root + 'handcraft/ddata1_woPCA.pickle'
    config.CT_mask = 'lung'
    config.CT_mask_type = 'all'
    config.remove_items = remove_items
    config.focal_alpha = [1, 1]
    config.focal_gamma = 2
    config.age_boundary = [1.04, 1.3]
    config.age_split_rate = 0.8

    # normalization and augmentation setting
    config.normalize = 'Z-Score_indicator-wise'
    config.normalize_concated = False
    config.CTs_augmentation = True
    config.crop_len = 64
    config.CTs_crop = False 
    config.UDA = False
    config.split_idx=[]

    # 3d Resnet config for CT scans encoder
    config.model_depth = 10 # model_depth in [10, 18, 34, 50, 101, 152, 200]
    config.n_input_channels = 1
    config.resnet_shortcut = 'B'
    config.conv1_t_size = 7
    config.conv1_t_stride = 2
    config.no_max_pool = True
    config.resnet_widen_factor = 1.0
    config.model = 'resnet'
    config.n_classes = 2

    check_dir = time.strftime('%Y-%m-%d %H:%M:%S')
    check_dir = 'Train_on_data1'+'_'+check_dir
    os.mkdir(os.path.join('checkpoints', check_dir))
    os.mkdir(os.path.join('res', check_dir))
    
    # loading raw clinical data
    config.data = config.data_root + 'data1'
    raw_data1 = load_data(config, 'train')

    config.data = config.data_root + 'data2'
    raw_data2 = load_data(config, 'train')

    config.data = config.data_root + 'data3'
    raw_data3 = load_data(config, 'train')

    accs_a = np.zeros(config.repeat)
    sens_a = np.zeros(config.repeat)
    specs_a = np.zeros(config.repeat)
    df_acc_all = []

    data1_train_idx = load_data1_split_idx(split='train')
    data1_val_idx = load_data1_split_idx(split='val')
    data1_test_idx = load_data1_split_idx(split='test')
    data1_train_idx = np.concatenate((data1_train_idx,data1_val_idx,data1_test_idx),axis=0)

    src_train_data = {key:item[data1_train_idx] for key, item in raw_data1.items()}
    # 5-fold cross-validation, split by age
    kf = RepeatedKFold_class_wise_split_by_age(n_splits=5, n_repeats=1, random_state=0, age_split=config.age_boundary)
    age_idx=config.keep_items.tolist().index('年龄')
    kf_split = kf.split(src_train_data['data'][:,age_idx], src_train_data['labels'])

    for i, (idx_train, idx_val) in enumerate(kf_split):
        config.split_idx.append(idx_val)
        print('=== START training for fold %d ===' % i)

        train_data = {key:item[idx_train] for key, item in src_train_data.items()}
        val_data = {key:item[idx_val] for key, item in src_train_data.items()}

        train_data = data_normalize(config,train_data, method='train')
        val_data = data_normalize(config,val_data, method='test')

        # initialize training dataset and compute the mean and std of CT scans
        data_set = dataset(train_data, config)
        mean_std_dic = data_set.valid_CTs_mean_std()
        config.update(mean_std_dic)
 
        use_cuda = True
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        for data in [train_data,val_data]:
            data['labels'] = torch.from_numpy(data['labels']).type(LongTensor)
            data['data'] = torch.from_numpy(data['data']).type(FloatTensor)
        
        model = MedicNet(config)

        if use_cuda:    
            model = model.cuda()
        best_model_a, ret_dict = train(model, train_data, val_data, config, i)
