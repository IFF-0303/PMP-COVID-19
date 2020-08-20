import pickle
import numpy as np
import os
import torch
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from model import MedicNet
from torchnet import meter
from tqdm import tqdm
from utils import dataset, load_data, data_normalize

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

    print('Loading models and config file...')
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


def test(model, test_dataloader, config, model_num):

    with torch.no_grad():
        model.eval()
        score_list=[]
        comatrix_list=[]
        i=0
        data_name = 'test'
        
        for test_batch in tqdm(test_dataloader):
            
            class_test = test_batch['labels']
            test_batch['ct_mask'] = test_batch['ct_mask'].cuda()
            test_batch['CTs'] = test_batch['CTs'].cuda()
            test_class_pred, _ = model(test_batch)

            confusion_matrix = meter.ConfusionMeter(config.num_classes)
            confusion_matrix.add(test_class_pred.detach().cpu(), class_test.cpu())
            comatrix_list.append(confusion_matrix.value())

            if i ==0:
                test_scores = test_class_pred
                test_labels = class_test
            else:
                test_scores = torch.cat((test_scores,test_class_pred),axis=0)
                test_labels=torch.cat((test_labels,class_test),axis=0)

            i = i+1

        model.train()
        cm_all = np.array(comatrix_list).sum(axis=0)
        print('Testing on dataset2 using pre-trained model %d.' % (model_num))
        print('Testing confusion matrix: ')
        print(cm_all)

        acc_epoch = (float(cm_all[0, 0])+float(cm_all[1, 1]))/cm_all.sum()
        spec_epoch = float(cm_all[0, 0]) / (cm_all[0, 0] + cm_all[0, 1]) # TN：val_cm[0, 0], FP：val_cm[0, 1],
        sen_epoch = float(cm_all[1, 1]) / (cm_all[1, 0] + cm_all[1, 1])  # FN：val_cm[1, 0], TP：val_cm[1, 1]
        precision_epoch = float(cm_all[1, 1]) / (cm_all[0, 1] + cm_all[1, 1]) 
      
        print('Test accuracy on %s: %.3f' % (data_name, acc_epoch))
        print('Test sensitivity on %s: %.3f' % (data_name, sen_epoch))
        print('Test specificity on %s: %.3f' % (data_name, spec_epoch))
        print('Test precisionon on %s: %.3f' % (data_name, precision_epoch))

        config.test_scores = test_scores
        config.test_labels = test_labels

        with open('res/%s/test_data2_fold%d_res.pickle' % ('test_res_on_dataset2_'+check_dir, model_num), 'wb') as fp:
            pickle.dump(config, fp)

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
    print('Loading pretrained model ...')
    model1, model2, model3, model4, model5, config = load_checkpoint(self_pretrained_model_path)
    config.data_root = '/home/cfang/works/COVID-19/PMP/data/'

    check_dir = time.strftime('%Y-%m-%d %H:%M:%S')
    os.mkdir(os.path.join('res', 'test_res_on_dataset2_'+check_dir))
    
    # load clinical data
    config.data = config.data_root + 'data2'
    raw_data2 = load_data(config, 'train')

    accs_a = np.zeros(config.repeat)
    sens_a = np.zeros(config.repeat)
    specs_a = np.zeros(config.repeat)
    df_acc_all = []

    test_data = data_normalize(config,raw_data2, method='test')

    use_cuda = True
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    for data in [test_data]:
        data['labels'] = torch.from_numpy(data['labels']).type(LongTensor)
        data['data'] = torch.from_numpy(data['data']).type(FloatTensor)
    
    test_dataloader = DataLoader(dataset(test_data, config), batch_size=6, shuffle=False)

    test(model1, test_dataloader, config, model_num=1)
    test(model2, test_dataloader, config, model_num=2)
    test(model3, test_dataloader, config, model_num=3)
    test(model4, test_dataloader, config, model_num=4)
    test(model5, test_dataloader, config, model_num=5) 