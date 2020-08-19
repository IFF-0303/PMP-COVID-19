import pickle
import numpy as np
import os
import torch
from model import MedicNet, MultiFocalLoss, Compute_class_score
from copy import deepcopy
from torchnet import meter
import torch.optim as optim
import torch.nn as nn
import math
import time
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pdb
from tqdm import tqdm
import time
import datetime
from torch.utils.data.sampler import WeightedRandomSampler
from warmup_scheduler import GradualWarmupScheduler
from utils import RepeatedKFold_class_wise, dataset, split_data, register, load_split_idx, \
                  random_subset, load_data, split, data_normalize, split_by_types, \
                  split_by_centers, data_augmentation, split_by_label, CTs_augmentation, \
                  load_data1_split_idx, CTs_augmentation_batch, compute_class_vector, \
                  seed_torch, label_smoothing, RepeatedKFold_class_wise_split_by_age
from sample_stratified import sample_stratified


def load_checkpoint(path):
    model1_file = sorted(os.listdir(path))[1]
    model2_file = sorted(os.listdir(path))[2]
    model3_file = sorted(os.listdir(path))[3]
    model4_file = sorted(os.listdir(path))[4]
    model5_file = sorted(os.listdir(path))[5]
    config_file = sorted(os.listdir(path))[0]
    
    model1_file = os.path.join(path, model1_file)
    model2_file = os.path.join(path, model2_file)
    model3_file = os.path.join(path, model3_file)
    model4_file = os.path.join(path, model4_file)
    model5_file = os.path.join(path, model5_file)
    config_file = os.path.join(path, config_file)

    print('Load Models and Config file...')
    print(model1_file)
    print(model2_file)
    print(model3_file)
    print(model4_file)
    print(model5_file)
    print(config_file)
    
    with open(config_file, 'rb') as fp:
        config = pickle.load(fp)

    model1 = MedicNet(config)
    model2 = MedicNet(config)
    model3 = MedicNet(config)
    model4 = MedicNet(config)
    model5 = MedicNet(config)

    model1.load_state_dict(torch.load(model1_file))
    model2.load_state_dict(torch.load(model2_file))
    model3.load_state_dict(torch.load(model3_file))
    model4.load_state_dict(torch.load(model4_file))
    model5.load_state_dict(torch.load(model5_file))

    return model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda(), model5.cuda(), config

# def test_WO_meta(model, support_data, test_data, config, num_fold):
    
#     support_dataloader = DataLoader(dataset(support_data, config), batch_size=config.few_shot_num*2, shuffle=False)
#     test_dataloader = DataLoader(dataset(test_data, config), batch_size=config.batch_size, shuffle=False)
    
#     comatrix_list=[]
#     ret_dict = {'accs': -1, 'sens':-1, 'specs':-1, 'precisions':-1, 'preds':None, 'labels':None}
#     data_name = 'data3'
    
#     with torch.no_grad():
#         model.eval()
#         compute_class_score.eval()

#         for support_data in tqdm(support_dataloader):

#             support_class = support_data['labels']
#             support_data['ct_mask'] = support_data['ct_mask'].cuda()
#             support_data['CTs'] = support_data['CTs'].cuda()

#             _, support_embed = model(support_data)
            
#             class_avg_vectors = compute_class_vector(support_embed, support_class, config.num_classes)

#         i = 0
#         for test_data in tqdm(test_dataloader):

#             class_test = test_data['labels']
#             test_data['ct_mask'] = test_data['ct_mask'].cuda()
#             test_data['CTs'] = test_data['CTs'].cuda()

#             test_class_pred, test_embed = model(test_data)
#             confusion_matrix = meter.ConfusionMeter(config.num_classes)
#             confusion_matrix.add(test_class_pred.detach().cpu(), class_test.cpu())
#             comatrix_list.append(confusion_matrix.value())

#             if i == 0:
#                 test_scores = test_class_pred
#                 test_labels = class_test
#             else:
#                 # pdb.set_trace()
#                 test_scores = torch.cat((test_scores,test_class_pred),axis=0)
#                 test_labels=torch.cat((test_labels,class_test),axis=0)

#             i = i+1


#         cm_all = np.array(comatrix_list).sum(axis=0)
#         print('%s test confusion matrix by metric-learning' % (data_name))
#         print(cm_all)

#         acc_fold = (float(cm_all[0, 0])+float(cm_all[1, 1]))/cm_all.sum()
#         spec_fold = float(cm_all[0, 0]) / (cm_all[0, 0] + cm_all[0, 1]) # TN：val_cm[0, 0], FP：val_cm[0, 1],
#         sen_fold = float(cm_all[1, 1]) / (cm_all[1, 0] + cm_all[1, 1])  # FN：val_cm[1, 0], TP：val_cm[1, 1]
#         precision_fold = float(cm_all[1, 1]) / (cm_all[0, 1] + cm_all[1, 1]) 

#         ret_dict['sens'] = sen_fold
#         ret_dict['specs'] = spec_fold
#         ret_dict['accs'] = acc_fold
#         ret_dict['precisions'] = precision_fold            
#         ret_dict['preds'] = test_scores.detach().cpu().numpy()
#         ret_dict['labels'] = test_labels.cpu().numpy()
      
#         print('%s acc: %.3f' % (data_name, acc_fold))
#         print('%s sen: %.3f' % (data_name, sen_fold))
#         print('%s spec: %.3f' % (data_name, spec_fold))
#         print('%s precisions: %.3f' % (data_name, precision_fold))

#         # save config
#         with open('checkpoints/%s/config.pickle' % check_dir, 'wb') as fp:
#             pickle.dump(config, fp)

#         saved_val = register(dump_file='./spss/test_data3_metric_learning_acc%.3f-sen%.3f_%s-focal_alpha%s-config.focal_gamma%d-bs%d-sdata_pool_%s-ddata_pool_%s-UDA-%s_CT_mask-%sepoch-%d-mask_type_%s.pickle' % \
#             (acc_fold, sen_fold, config.encoder_3d, config.focal_alpha,config.focal_gamma,config.batch_size,config.sdata_pool,config.ddata_pool,config.UDA,config.CT_mask,config.max_epoch,config.CT_mask_type))

#         saved_val.regis(ret_dict)
#         saved_val.dump()

def test(model, train_class0_dataloader, train_class1_dataloader, test_dataloader, config, epoch, best_acc, best_sen, best_val_epoch, best_model_state_dict, ret_dict, num_fold):
    
    # support_dataloader = DataLoader(dataset(support_data, config), batch_size=config.few_shot_num*2, shuffle=False)
    # test_dataloader = DataLoader(dataset(test_data, config), batch_size=config.batch_size, shuffle=False)
    
    comatrix_list=[]
    ret_dict = {'accs': -1, 'sens':-1, 'specs':-1, 'precisions':-1, 'preds':None, 'labels':None}
    support_data = {'labels': -1, 'data':-1, 'order':-1, 'domain':-1, 'CTs':-1, 'ct_mask':-1}
    data_name = 'dataset3'
    # best_model_state_dict = model.state_dict()
    
    # best_acc = -1
    # best_sen = -1
    # ret_dict = {'accs': -1, 'sens':-1, 'specs':-1, 'precisions':-1, 'preds':None, 'labels':None}
    # best_val_epoch = 0
    with torch.no_grad():
        model.eval()
        compute_class_score.eval()

        data_zip = enumerate(zip(train_class0_dataloader, train_class1_dataloader))
        for step, (data_class0, data_class1) in data_zip:


            for key in data_class0.keys():
                # support_data[key] = torch.cat((data_class0[key][:-1], data_class1[key][:-1]), 0)
                # support_data[key] = torch.cat((data_class0[key][1:], data_class1[key][1:]), 0)
                support_data[key] = torch.cat((data_class0[key], data_class1[key]), 0)

            support_class = support_data['labels']
            support_data['ct_mask'] = support_data['ct_mask'].cuda()
            support_data['CTs'] = support_data['CTs'].cuda()

            _, support_embed = model(support_data)

            # support_label_class0 = data_class0['labels']
            # data_class0['ct_mask'] = data_class0['ct_mask'].cuda()
            # data_class0['CTs'] = data_class0['CTs'].cuda()

            # support_label_class1 = data_class1['labels']
            # data_class1['ct_mask'] = data_class1['ct_mask'].cuda()
            # data_class1['CTs'] = data_class1['CTs'].cuda()

            # _, support_class0_embed = model(data_class0)
            # _, support_class1_embed = model(data_class1)

            # support_embed = torch.cat((support_class0_embed,support_class1_embed),axis=0)

            # support_class = torch.cat((support_label_class0,support_label_class1),axis=0)
            
            class_avg_vectors = compute_class_vector(support_embed, support_class, config.num_classes)



        # data_zip = enumerate(zip(support_dataloader, test_dataloader))
        # for step, (support_data, test_data) in data_zip:

        i = 0
        for test_data in tqdm(test_dataloader):

            class_test = test_data['labels']
            test_data['ct_mask'] = test_data['ct_mask'].cuda()
            test_data['CTs'] = test_data['CTs'].cuda()

            _, test_embed = model(test_data)
            # pdb.set_trace()

            class_score, test_tau = compute_class_score(class_avg_vectors[0].unsqueeze(0).repeat(test_embed.shape[0],1), class_avg_vectors[1].unsqueeze(0).repeat(test_embed.shape[0],1), test_embed)
            # pdb.set_trace()
            confusion_matrix = meter.ConfusionMeter(config.num_classes)
            confusion_matrix.add(class_score.detach().cpu(), class_test.cpu())
            comatrix_list.append(confusion_matrix.value())

            if i == 0:
                scores = class_score
                labels = class_test
            else:
                # pdb.set_trace()
                scores = torch.cat((scores,class_score),axis=0)
                labels = torch.cat((labels,class_test),axis=0)

            i = i+1

        model.train()
        compute_class_score.train()

        print('Epoch ', epoch+1)
        print('test tau: ',test_tau)
        cm_all = np.array(comatrix_list).sum(axis=0)
        print('test confusion matrix on %s by metric-learning' % (data_name))
        print(cm_all)

        acc_epoch = (float(cm_all[0, 0])+float(cm_all[1, 1]))/cm_all.sum()
        spec_epoch = float(cm_all[0, 0]) / (cm_all[0, 0] + cm_all[0, 1]) # TN：val_cm[0, 0], FP：val_cm[0, 1],
        sen_epoch = float(cm_all[1, 1]) / (cm_all[1, 0] + cm_all[1, 1])  # FN：val_cm[1, 0], TP：val_cm[1, 1]
        precision_epoch = float(cm_all[1, 1]) / (cm_all[0, 1] + cm_all[1, 1]) 

        ret_dict['sens'] = sen_epoch
        ret_dict['specs'] = spec_epoch
        ret_dict['accs'] = acc_epoch
        ret_dict['precisions'] = precision_epoch            
        ret_dict['preds'] = scores.detach().cpu().numpy()
        ret_dict['labels'] = labels.cpu().numpy()
      
        print('Test accuracy on %s: %.3f' % (data_name, acc_epoch))
        print('Test sensitivity on %s: %.3f' % (data_name, sen_epoch))
        print('Test specificity on %s: %.3f' % (data_name, spec_epoch))
        print('Test precisionon on %s: %.3f' % (data_name, precision_epoch))

        if acc_epoch>=best_acc:
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
            root = './checkpoints/%s/%d' % (check_dir, num_fold)
            if not os.path.exists(root):
                os.mkdir(root)
                
            torch.save(best_model_state_dict, '%s/acc%.3f_sen%.3f_@epoch%d_bs%d-sdata_pool_%s-ddata_pool_%s-UDA-%s_CT_mask-%s-%s_normalize-%s.pth' % \
                (root, best_acc,best_sen,epoch+1,config.batch_size,config.sdata_pool, \
                    config.ddata_pool,config.UDA,config.CT_mask,config.CT_mask_type,config.normalize))
            
            # save config
            with open('checkpoints/%s/config.pickle' % check_dir, 'wb') as fp:
                pickle.dump(config, fp)

            saved_val = register(dump_file='./spss/metric_learning_res_on _dataset3/acc%.3f-sen%.3f_%s@epoch%d_bs%d-sdata_pool_%s-ddata_pool_%s-UDA-%s_CT_mask-%sepoch-%d-mask_type_%s.pickle' % \
                (best_acc, best_sen, config.encoder_3d, epoch+1, config.batch_size,config.sdata_pool,config.ddata_pool,config.UDA,config.CT_mask,config.max_epoch,config.CT_mask_type))

            saved_val.regis(ret_dict)
            saved_val.dump()

    return best_model_state_dict, best_acc, best_sen, best_val_epoch, ret_dict

        # save config
        # with open('checkpoints/%s/config.pickle' % check_dir, 'wb') as fp:
        #     pickle.dump(config, fp)

        # saved_val = register(dump_file='./spss/data1_meta_learning/acc%.3f-sen%.3f_%s-focal_alpha%s-config.focal_gamma%d-bs%d-sdata_pool_%s-ddata_pool_%s-UDA-%s_CT_mask-%sepoch-%d-mask_type_%s.pickle' % \
        #     (acc_fold, sen_fold, config.encoder_3d, config.focal_alpha,config.focal_gamma,config.batch_size,config.sdata_pool,config.ddata_pool,config.UDA,config.CT_mask,config.max_epoch,config.CT_mask_type))

        # saved_val.regis(ret_dict)
        # saved_val.dump()

def train(model, train_data_class0, train_data_class1, test_data, config, num_fold):
    
    train_class0_dataloader = DataLoader(dataset(train_data_class0, config), batch_size=config.few_shot_num, shuffle=True)
    train_class1_dataloader = DataLoader(dataset(train_data_class1, config), batch_size=config.few_shot_num, shuffle=True)
    test_dataloader = DataLoader(dataset(test_data, config), batch_size=config.batch_size, shuffle=False)
    
    # for param in model.classifier_final.parameters():
    #     param.requires_grad = True

    # for param in model.layer3.parameters():
    #     param.requires_grad = True

    # for param in model.encoder_3d.layer3.parameters():
    #     param.requires_grad = True

    

    # for param in model.lstm.parameters():
    #     param.requires_grad = True
    
    # optimizer = optim.Adam(model.parameters(),lr=config.lr, weight_decay = 1e-4)
    optimizer = optim.Adam([
    # optimizer = optim.SGD([
                    {'params': model.parameters()},
                    {'params': compute_class_score.parameters(), 'lr': config.tau_lr}
                ], lr=config.lr, weight_decay = 1e-4)
                # ], lr=config.lr, momentum=0.9, weight_decay = 1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, nesterov=True)
    # scheduler_StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1) 
    # scheduler_CA = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_train_epoch, eta_min=1e-06)
    # scheduler_MultiStep = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,7,9], gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_MultiStep)
    # scheduler_CAWR = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1) 
    # T_0: Number of epochs for the first restart
    # T_mult: A factor increases Ti after a restart. Default: 1.

    # if config.enable_focal_loss == True:
    #     criterion = MultiFocalLoss(config.num_classes, alpha=config.focal_alpha, gamma=config.focal_gamma)
    # else:
    #     criterion_CE = nn.CrossEntropyLoss()

    BCE_criterion = nn.BCELoss()
    cos_criterion = nn.CosineEmbeddingLoss(margin=config.cos_loss_margin)
    
    best_model_state_dict = model.state_dict()
    
    best_acc = -1
    best_sen = -1
    ret_dict = {'accs': -1, 'sens':-1, 'specs':-1, 'precisions':-1, 'preds':None, 'labels':None}
    support_data = {'labels': -1, 'data':-1, 'order':-1, 'domain':-1, 'CTs':-1, 'ct_mask':-1}
    query_data = {'labels': -1, 'data':-1, 'order':-1, 'domain':-1, 'CTs':-1, 'ct_mask':-1}
    best_val_epoch = 0

    # iters = len(train_dataloader)

    for epoch in range(config.finetune_epoch):

        data_zip = enumerate(zip(train_class0_dataloader, train_class1_dataloader))
        for step, (data_class0, data_class1) in data_zip:

            for key in data_class0.keys():
                support_data[key] = torch.cat((data_class0[key][:-1], data_class1[key][:-1]), 0)
                # support_data[key] = torch.cat((data_class0[key][1:], data_class1[key][1:]), 0)
                query_data[key] = torch.cat((data_class0[key][-1].unsqueeze(0), data_class1[key][-1].unsqueeze(0)), 0)
                # query_data[key] = torch.cat((data_class0[key][0].unsqueeze(0), data_class1[key][0].unsqueeze(0)), 0)

            if config.CTs_augmentation==True:
                support_data = CTs_augmentation_batch(config, support_data)
                query_data = CTs_augmentation_batch(config, query_data)

            support_class = support_data['labels']
            support_data['ct_mask'] = support_data['ct_mask'].cuda()
            support_data['CTs'] = support_data['CTs'].cuda()

            query_class = query_data['labels']
            query_data['ct_mask'] = query_data['ct_mask'].cuda()
            query_data['CTs'] = query_data['CTs'].cuda()

            optimizer.zero_grad()

            support_class_pred, support_embed = model(support_data)
            query_class_pred, query_embed = model(query_data)

            class_avg_vectors = compute_class_vector(support_embed, support_class, config.num_classes)
            query_class_score, _ = compute_class_score(class_avg_vectors[0].unsqueeze(0).repeat(query_embed.shape[0],1), class_avg_vectors[1].unsqueeze(0).repeat(query_embed.shape[0],1), query_embed)

            # print(class_avg_vectors)
            # pdb.set_trace()
            # print('query_class_score: ', query_class_score)
            # print('query_class: ', query_class)
            # pdb.set_trace()
            y = (torch.ones(1)*(-1)).cuda()
            metric_loss = cos_criterion(class_avg_vectors[0].unsqueeze(0),class_avg_vectors[1].unsqueeze(0),y.unsqueeze(0))
            class_loss = BCE_criterion(query_class_score[:,1], query_class.float())
            loss = class_loss + config.cos_loss_gamma * metric_loss
            print('class_loss %.3f metric_loss %.3f' % (class_loss, metric_loss))
            loss.backward()
            optimizer.step()

        best_model_state_dict, best_acc, best_sen, best_val_epoch, ret_dict = test(model, train_class0_dataloader, train_class1_dataloader, test_dataloader, config, epoch, best_acc, best_sen, best_val_epoch, best_model_state_dict, ret_dict, num_fold)
        print('Accuracy %.3f sensitivity %.3f @ epoch %d' % (best_acc,best_sen,best_val_epoch+1))
        # scheduler_StepLR.step()
        print('current lr: ', optimizer.param_groups[0]['lr'])


if __name__ == "__main__":

    remove_items = ['医务人员接触感染', '心梗', '脑梗', '心脑血管病',
    '慢性消化系统疾病','外周血管病', '慢性呼吸道疾病', 
    '自身免疫或结缔组织病', '服用糖皮质激素强的松15mgd×30天以上', 
    '发病第几天出现气喘','咯血','ThTs','淋巴细胞CD45CD45abs', 
    '咳嗽','HCRP', 'PCT','肌钙蛋白','BNP'
    ]

    config = edict()
    # path setting
    self_pretrained_model_path = '/home/cfang/works/COVID-19/PMP/code/checkpoints/pretrained_models_on_dataset1_5_fold'
    print('Loading pre-trained models ...')
    model1, model2, model3, model4, model5, config = load_checkpoint(self_pretrained_model_path)
    config.data_root = '/home/cfang/works/COVID-19/PMP/data/'
    
    
    # finetune config
    config.finetune_repeat = 2
    config.finetune_epoch = 100
    config.batch_size = 20
    config.few_shot_num = 10
    # config.tau = 3
    config.init_tau = 5
    config.tau_lr = 0.1
    config.lr = 0.01
    config.cos_loss_margin = 0.2
    config.cos_loss_gamma = 0.5
    config.seed = 8888
    seed_torch(seed=config.seed)
    

    print(config)

    # 加载原始临床数据
    config.data = config.data_root + 'data3'
    raw_data3 = load_data(config, 'train')

    # config.data = config.data_root + 'data2'
    # raw_data2 = load_data(config, 'train')

    # config.data = config.data_root + 'data3'
    # raw_data3 = load_data(config, 'train')

    # saved_val = register(dump_file='./spss/val_data3-train8-val2_%s-focal_alpha%s-config.focal_gamma%d-bs%d-sdata_pool_%s-ddata_pool_%s-UDA-%s_CT_mask-%sepoch-%d-mask_type_%s.pickle' % (config.encoder_3d, config.focal_alpha,config.focal_gamma,config.batch_size,config.sdata_pool,config.ddata_pool,config.UDA,config.CT_mask,config.max_epoch,config.CT_mask_type))

    train_start_time = time.time()

    # for test_data_flag in range(1,-1,-2): # train for data2
    for test_data_flag in range(1): # train for data3

        for finetune_repeat in range(config.finetune_repeat):
            check_dir = time.strftime('%Y-%m-%d %H:%M:%S')
            check_dir = 'data3_query-1_seed'+str(config.seed)+'_tau'+str(config.init_tau)+'_epoch'+str(config.finetune_epoch)+'_tau'+str(config.init_tau)+'_margin'+str(config.cos_loss_margin)+'_gamma'+str(config.cos_loss_gamma)+'_'+check_dir+'_repeat_'+str(finetune_repeat)+'_few_shot_'+str(config.few_shot_num)
            os.mkdir(os.path.join('checkpoints', check_dir))

            # 获取训练和测试数据索引
            if test_data_flag==0:
                
                data3_idx = []
                data3_test_idx = []

                age_idx=config.keep_items.tolist().index('年龄')
                data3_train_idx0, data3_train_idx1 = sample_stratified(config.few_shot_num, raw_data3, age_idx, config.age_boundary)
                data3_train_idx = np.concatenate((data3_train_idx0,data3_train_idx1),axis=0)


                for idx in range(raw_data3['data'].shape[0]):
                    data3_idx.append(idx)

                for idx in data3_idx:
                    if idx not in data3_train_idx:  
                        data3_test_idx.append(idx)

                
                train_data_class0 = {key:item[data3_train_idx0] for key, item in raw_data3.items()}
                train_data_class1 = {key:item[data3_train_idx1] for key, item in raw_data3.items()}
                test_data = {key:item[data3_test_idx] for key, item in raw_data3.items()}

            elif test_data_flag==1:
                
                data2_idx = []
                data2_test_idx = []

                
                age_idx=config.keep_items.tolist().index('年龄')
                data2_train_idx0, data2_train_idx1 = sample_stratified(config.few_shot_num, raw_data2, age_idx, config.age_boundary)
                data2_train_idx = np.concatenate((data2_train_idx0,data2_train_idx1),axis=0)


                for idx in range(raw_data2['data'].shape[0]):
                    data2_idx.append(idx)

                for idx in data2_idx:
                    if idx not in data2_train_idx:  
                        data2_test_idx.append(idx)

                train_data_class0 = {key:item[data2_train_idx0] for key, item in raw_data2.items()}
                train_data_class1 = {key:item[data2_train_idx1] for key, item in raw_data2.items()}
                test_data = {key:item[data2_test_idx] for key, item in raw_data2.items()}


            accs_a = np.zeros(config.repeat)
            sens_a = np.zeros(config.repeat)
            specs_a = np.zeros(config.repeat)
            df_acc_all = []

            train_data_class0 = data_normalize(config,train_data_class0, method='test')
            train_data_class1 = data_normalize(config,train_data_class1, method='test')
            test_data = data_normalize(config,test_data, method='test')

            use_cuda = True
            FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

            for data in [train_data_class0, train_data_class1, test_data]:
                data['labels'] = torch.from_numpy(data['labels']).type(LongTensor)
                data['data'] = torch.from_numpy(data['data']).type(FloatTensor)

            # train_dataloader = DataLoader(dataset(train_data, config), batch_size=32, shuffle=True)
            # test_dataloader = DataLoader(dataset(test_data, config), batch_size=32, shuffle=False)
            for i in range(5):
                if i == 0:
                    model = model3
                elif i ==1:
                    model = model2
                elif i ==2:
                    model = model1
                elif i ==3:
                    model = model4
                elif i ==4:
                    model = model5

                # for param in model.parameters():
                #     param.requires_grad = False

                compute_class_score = Compute_class_score(tau=config.init_tau)
                compute_class_score = compute_class_score.cuda()

                train(model, train_data_class0, train_data_class1, test_data, config, i)
                # test_WO_meta(model, train_data, test_data, config, i) # baseline

    print('Training time: ', datetime.timedelta(seconds=(time.time()-train_start_time)))