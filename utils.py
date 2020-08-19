import pickle
import numpy as np
import os
import torch
import imageio
import random
import csv

from copy import deepcopy
from sklearn.model_selection import KFold, RepeatedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from sklearn import preprocessing
from tqdm import tqdm
from functools import partialmethod
from sklearn.metrics import precision_recall_fscore_support

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def label_smoothing(label, epsilon=0.05):
    label = label.float()
    mask0 = (label==0)
    label[mask0] = epsilon
    label[~mask0] = 1-epsilon

    return label

def plot_scan(scan,i,scan_number):
    temp = '{}_{}.png'.format(scan_number, i+1)
    imageio.imwrite(os.path.join('./temp/', temp), scan)

def plot_CT_scans(data,scan_number):
    sequence_len = data.shape[0]
    for i in range(sequence_len):
        plot_scan(data[i],i,scan_number)
 
class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        dataA, dataB = None, None
        try:
            dataA = next(self.data_loader_A_iter)
        except StopIteration:
            
            self.stop_A = True
            self.data_loader_A_iter = iter(self.data_loader_A)
            dataA = next(self.data_loader_A_iter)

        try:
            dataB = next(self.data_loader_B_iter)
        except StopIteration:
            
            self.stop_B = True
            self.data_loader_B_iter = iter(self.data_loader_B)
            dataB = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            keys = list(dataA.keys())
            # concatenate two dictionaries
            return {k: torch.cat([dataA.pop(k), dataB.pop(k)], dim=0) for k in keys}

def compute_class_vector(vectors, labels, num_classes):

    mask = []
    class_avg_vectors = torch.zeros((num_classes,vectors.shape[1])).cuda()
    for i in range(num_classes):
        mask.append(labels==i)
    for i in range(num_classes):
        class_avg_vectors[i] = vectors[mask[i]].mean(axis=0)

    return class_avg_vectors

def CTs_augmentation_batch(config, data):
    batch_shape=data['CTs'].shape
    augmented_batch=torch.zeros((batch_shape[0],batch_shape[1],batch_shape[2],config.crop_len,config.crop_len))

    for k in range(7):
        for j in range(data['CTs'].shape[0]):
            aug_mode_1=str(random.sample(range(0,3),1)[0])
            aug_mode_2=str(random.sample(range(0,4),1)[0])
            aug_mode = aug_mode_1+aug_mode_2
            augmented_batch[j,:,k,:,:]=torch.from_numpy(CTs_augmentation(data['CTs'][j,:,k,:,:].cpu().numpy(), aug_mode, crop=config.CTs_crop, crop_len=config.crop_len))

    data['CTs']=augmented_batch.cuda()

    return data

def random_xy_crop_for_3D_img(img, crop_y, crop_x):
    # fixed size random clipping in xy plane
    crop_xmin=random.sample(range(0,img.shape[2]-crop_x+1),1)[0]
    crop_ymin=random.sample(range(0,img.shape[1]-crop_y+1),1)[0]
    img = img[:, crop_ymin:(crop_ymin+crop_y), crop_xmin:(crop_xmin+crop_x)]
    return img

def clinical_augmentation(data, p=0.5):
    # randomly zeroes some of the elements of the input tensor with probability p
    batch_size = data.size()[0]
    batch_size = data.size()[0]
    return data_augmented

def CTs_augmentation(data, mode, crop, crop_len):
    # mode 00: original
    # mode 10: fliplr
    # mode 20: flipup
    # mode 01: rot90
    # mode 11: fliplr and rot90
    # mode 21: flipup and rot90
    # mode 02: rot180
    # mode 12: fliplr and rot180
    # mode 22: flipup and rot180
    # mode 03: rot270
    # mode 13: fliplr and rot270
    # mode 23: flipup and rot270
    if mode == '00':
        # original
        pass
        
    elif mode == '10':
        # fliplr
        for i in range(data.shape[0]):
            data[i]=np.fliplr(data[i])
     
    elif mode == '20':
        # flipup
        for i in range(data.shape[0]):
            data[i]=np.flipud(data[i])

    elif mode == '01':
        # rot90
        for i in range(data.shape[0]):
            data[i]=np.rot90(data[i])

    elif mode == '11':
        # fliplr and rot90
        for i in range(data.shape[0]):
            data[i]=np.fliplr(np.rot90(data[i]))

    elif mode == '21':
        # flipup and rot90
        for i in range(data.shape[0]):
            data[i]=np.flipud(np.rot90(data[i]))

    elif mode == '02':
        # rot180
        for i in range(data.shape[0]):
            data[i]=np.rot90(data[i], k=2)

    elif mode == '12':
        # fliplr and rot180
        for i in range(data.shape[0]):
            data[i]=np.fliplr(np.rot90(data[i], k=2))

    elif mode == '22':
        # flipup and rot180
        for i in range(data.shape[0]):
            data[i]=np.flipud(np.rot90(data[i], k=2))

    elif mode == '03':
        # rot270
        for i in range(data.shape[0]):
            data[i]=np.rot90(data[i], k=3)

    elif mode == '13':
        # fliplr and rot270
        for i in range(data.shape[0]):
            data[i]=np.fliplr(np.rot90(data[i], k=3))

    elif mode == '23':
        # flipup and rot270
        for i in range(data.shape[0]):
            data[i]=np.flipud(np.rot90(data[i], k=3))

    if crop==False:
        return data
    elif crop==True:
        return random_xy_crop_for_3D_img(data,crop_len,crop_len)

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min+1e-6)

def clinical_data_normalization(data,norm_type='minmax'):
    data_min = data.min()
    data_max = data.max()
    data_mean = data.mean()
    data_std = data.std()
    if norm_type == 'minmax':
        return (data - data_min)/(data_max-data_min+1e-6)
    elif norm_type == 'mean':
        return (data - data_mean)/(data_max-data_min+1e-6)
    elif norm_type == 'z-score':
        return (data - data_mean)/(data_max-data_min+1e-6)
    elif norm_type == 'l1':
        return data/(np.linalg.norm(data,ord=1)+1e-6)
    elif norm_type == 'l2':
        return data/(np.linalg.norm(data,ord=2)+1e-6)

def data_augmentation(data, repeat_number=5, shuffle=True):
    ori_data = data['data']
    ori_label = data['labels']
    ori_code = data['codes']
    ori_type = data['types']
    for i in range(repeat_number-1):
        data['data']=np.append(data['data'], ori_data*(1+(np.random.rand()-0.5)*0.1),axis=0)
        data['labels']=np.append(data['labels'],ori_label,axis=0)
        data['codes']=np.append(data['codes'],ori_code,axis=0)
        data['types']=np.append(data['types'],ori_type,axis=0)
        
    if shuffle:
        permutation = list(np.random.permutation(data['data'].shape[0]))
        data['data'] = data['data'][permutation, :]
        data['labels'] = data['labels'][permutation]
        data['codes'] = data['codes'][permutation]
        data['types'] = data['types'][permutation]
 
    return data

class RepeatedKFold_class_wise(RepeatedKFold):
    # 5 fold sampling
    def __init__(self, n_splits=5, n_repeats=10, random_state=None, extra=['test', None]):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits

    def split(self, X, y):
        n_class = set(y).__len__()
        generators_idx = []
        index = []
        
        for i in range(n_class):
            index.append(np.arange(X.shape[0])[y==i])
            X_i = X[y==i]
            generators_idx.append(super().split(X_i))
            
        for i in range(self.n_repeats*self.n_splits):
            
            idx = [next(subidx) for subidx in generators_idx]
            idx_train = np.concatenate([index[i][subidx[0]] for i, subidx in enumerate(idx)])
            idx_test = np.concatenate([index[i][subidx[1]] for i, subidx in enumerate(idx)])

            yield idx_train, idx_test

class RepeatedKFold_class_wise_split_by_age(RepeatedKFold):
    # split by age
    def __init__(self, n_splits=5, n_repeats=10, random_state=None, extra=['test', None], age_split=[]):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits
        self.age_split = age_split

    def age_stratified(self, X, y):
        """
        X is column vector of age
        y is class label
        """
        land = np.logical_and
        lor = np.logical_or
        left = -1000
        ret_mask = []
        for right in self.age_split:
            m = land(X > left, X < right+0.0001)
            ret_mask.append(m[:, None]) 
            left = right
        m = X > left
        ret_mask.append(m[:, None])
        mask = np.concatenate(ret_mask, axis=1)
        m = y == 1
        temp = deepcopy(mask)
        mask[m] = False
        temp[~m] = False
        mask = np.concatenate([mask, temp], axis=1)
        ind = np.arange(mask.shape[1])[None,].repeat(mask.shape[0], axis=0)
        stratified_ind = ind[mask]
        return stratified_ind
    
    def split(self, X, y):
        n_class = set(y).__len__()
        assert n_class == 2
        generators_idx = []
        index = []
        ind = self.age_stratified(X, y)
        ind_set = set(ind)
        for i in ind_set:
            index.append(np.arange(X.shape[0])[ind==i])
            X_i = X[ind==i]
            generators_idx.append(super().split(X_i))
            
        for i in range(self.n_repeats*self.n_splits):
            idx = [next(subidx) for subidx in generators_idx]
            idx_train = np.concatenate([index[i][subidx[0]] for i, subidx in enumerate(idx)])
            idx_test = np.concatenate([index[i][subidx[1]] for i, subidx in enumerate(idx)])

            yield idx_train, idx_test

### ========= Define data augmentation methods =========

class Random_zero(object):
    '''
    Random zero
    n_zero : the number of random zero
    p      : the probability of random zero
    '''
    def __init__(self, param_dic):
        self.n_zero = param_dic['n_zero']
        self.p = param_dic['p']

    def __call__(self, vec):
        n_choose = self.n_zero if self.n_zero > 0 else len(vec)
        idx = np.random.choice(range(len(vec)), n_choose, replace=False)
        idx = [num for num in idx if np.random.rand()<=self.p]
        vec[idx] = 0.
        return vec

class Random_scale(object):
    '''
    Random scaling
    p: the probability of scaling
    s: the scaling rate
    '''
    def __init__(self, param_dic):
        self.p = param_dic['p']
        self.s = param_dic['s']

    def __call__(self, vec):
        if np.random.rand() <= self.p:
            vec = torch.tensor(self.s).type_as(vec) * vec
        return vec

class Random_noise(object): 
    '''
    random noise
    p: the probability of adding noise
    sigma: the scale of noise
    '''
    def __init__(self, param_dict):
        self.p = param_dict['p']
        self.sigma = param_dict['sigma']

    def __call__(self, vec):
        if np.random.rand() < self.p:
            tt= torch.tensor(self.sigma) * torch.randn(vec.shape)
            tt = tt.type_as(vec)
            vec = tt + vec
        return vec

class Identity(object):
    '''
    no transformation, for debugging
    '''
    def __init__(self, param_dict):
        pass

    def __call__(self, vec):
        return vec

mapping = {'Random_zero':  Random_zero,
           'Random_scale': Random_scale,
           'Random_noise': Random_noise,
           'Identity':     Identity}

class dataset(Dataset):
    def __init__(self, data, config, only_sdata=False):
        super().__init__()
        self.data = data
        self.CT_mask = config.CT_mask
        self.CT_mask_type = config.CT_mask_type
        self.only_sdata = only_sdata
        self.CTs_mean = None if 'CTs_mean' not in config.keys() else config.CTs_mean
        self.CTs_std = None if 'CTs_std' not in config.keys() else config.CTs_std
        data_root = '/home/cfang/works/COVID-19/PMP/data/CT'
       
        if self.CT_mask == 'lung':
            sub_dir = 'CT_DATA_lung_mask'
        elif self.CT_mask == 'ncp':
            sub_dir = 'CT_DATA_ncp_mask'
        elif self.CT_mask == 'no':
            sub_dir = 'CT_DATA_64_64_64'
        
        self.pickle_dir = os.path.join(data_root, sub_dir)
        if self.CT_mask_type == 'all':
            self.mask_file = os.path.join(data_root, 'ct_mask.pkl')
        elif self.CT_mask_type == '15':
            self.mask_file = os.path.join(data_root, 'ct_mask_15.pkl')
        elif self.CT_mask_type == 'density':
            self.mask_file = os.path.join(data_root, 'ct_mask_density.pkl')
        self.valid = self.get_valid_mask()
    
    def set_CTs_mean_std(self, dic):
        self.CTs_mean = dic['CTs_mean']
        self.CTs_std = dic['CTs_std']
        
    def get_valid_mask(self):
        
        with open(self.mask_file, 'rb') as fp:
            valid = pickle.load(fp)
        return valid

    def __getitem__(self, idx):
        if self.CTs_mean is None or self.CTs_std is None:
            raise ValueError('Please compute CTs mean and std first BY \n 1. set set_CTs_mean_std \n 2. valid_CTs_mean_std')
   
        if self.only_sdata:
             ret = {key: self.data[key][idx] for key in self.data.keys()}
        else:
            ret = self.getitem_normalize(idx)
 
        return ret

    def load_CT_file(self, order): 
        """
        order is the patient number
        """    
        pickle_file = os.path.join(self.pickle_dir, '%d.pickle'%order)
        try:
            with open(pickle_file, 'rb') as fp:
                CTs = pickle.load(fp)
                
                CTs = CTs if order < 500 else CTs['3D_image']
        except FileNotFoundError:
            CTs = []
        return CTs
    
    def valid_CTs_mean_std(self):
        print('Start compute mean and std for all valid CT scans...')
        orders = self.data['order']
        means = []
        stds = []
        for idx in tqdm(range(orders.__len__())):
            data = self.getitem(idx)
            if data['ct_mask'].sum() < 1:
                continue
            imgs = data['CTs'][data['ct_mask']] 
            
            nonzero_index=imgs>0
        
            # Mean and STD were calculated in the lung area
            means.append(imgs[nonzero_index].mean().unsqueeze(0))
            stds.append(imgs[nonzero_index].std().unsqueeze(0))

        mean = torch.cat(means, dim=0).mean(dim=0)
        std = torch.cat(stds, dim=0).mean(dim=0)
        self.CTs_mean = mean
        self.CTs_std = std
        return {'CTs_mean':mean, 'CTs_std':std}
    
    def getitem_normalize(self, idx): 
        """
        1. zero for invalid CTs
        2. normalization for valid CTs
        """
        data = self.getitem(idx)
        m = data['ct_mask']
        nonzero_index=data['CTs'][m]>0
        data['CTs'][~m] = 0.
        data['CTs'][m] = (data['CTs'][m] - self.CTs_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / (self.CTs_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+1e-8)
            
        return data   
        
    def getitem(self, idx):        
        ret = {key: self.data[key][idx] for key in self.data.keys()}
        
        order = ret['order']
        valid_mask = np.zeros((7,)) == 1
        try:
            valid = int(self.valid[float(order)])
        except:
            valid = 7
        
        valid_mask[:valid] = True
        valid_mask = torch.from_numpy(valid_mask)

        CTs = self.load_CT_file(order)

        mask = torch.zeros((7,))
        mask[:len(CTs)] = 1
        mask = mask == 1   
        mask = mask & valid_mask
   
        CTs = [x.squeeze(0) for x in CTs]
        CTs += [torch.zeros((64, 64, 64))] * (7-len(CTs))
        CTs = [x.unsqueeze(0) for x in CTs]
        CTs = CTs[:7] # more than 7 CT scans are removed
        CTs = torch.cat(CTs, dim=0)
        CTs = self.img_tranform_3(CTs)
        ret['CTs'] = CTs
        ret['ct_mask'] = mask
        return ret

    def img_tranform_3(self, img):
        
        # window width and level of lung CT
        center = -500
        width = 1500
        
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)
        img = img - min
        img = torch.trunc( img * dFactor)
        img[img < 0.0] = 0
        img[img > 255.0] = 255
        
        img /= 255.
        return img

    def img_tranform(self, img):
        img[img==-3000] = 0
        return img

    def CTs_normalization(image_3d, norm_type='z-score'):
        for i in range(image_3d.shape[0]):
            if norm_type=='z-score':
                image_3d[i] = z_score_3d(image_3d[i].squeeze()).unsqueeze(0)

            elif norm_type=='zero-one':
                image_3d[i] = zero_one_scale_3d(image_3d[i].squeeze()).unsqueeze(0)

        return image_3d


    def transform(self, vec):
        T = [mapping[trans](param_dict) for trans, param_dict in self.t_dict.items()]
        return Compose(T)(vec)

    def __len__(self):
        return self.data['labels'].__len__()
    

def split_data(data, training_idx, testing_idx, use_cuda=True):
    testing_data = {}
    training_data = {}
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    for key in data.keys():
        testing_data[key] = torch.tensor(data[key][testing_idx])
        training_data[key] = torch.tensor(data[key][training_idx])
    
    for data in [testing_data, training_data]:
        data['labels'] = data['labels'].type(LongTensor)
        data['ddata'] = data['ddata'].type(FloatTensor)
        data['data'] = data['data'].type(FloatTensor)
    return training_data, testing_data

class register(object):
    def __init__(self, keys=['accs', 'sens', 'specs', 'precisions', 'labels', 'preds'], dump_file=None):
        self.dump_file = dump_file
        self.dic = {}
        for key in keys:
            self.dic[key] = []
        
    def regis(self, x):
        for key in x.keys():
            self.dic[key].append(x[key])

    def dump(self, file=None):
        if file is None:
            file = self.dump_file
        
        with open(file, 'wb') as fp:
            keys = list(self.dic.keys())
            pickle.dump(keys, fp)
            for key, value in self.dic.items():
                pickle.dump(value, fp)
          
        
def random_subset(data, num, balance=True):
    if num < 1:
        return data
    labels = data['labels']
    index = np.arange(labels.__len__())
    if balance:
        classes = set(labels)
        cnum = num // len(classes)
        choice = [np.random.choice(index[labels==i], cnum, replace=False) for i in classes]
        choice = np.array(choice).ravel().tolist()
    else:
        choice = np.random.choice(index, num, replace=False).tolist()

    for key in data.keys():
        data[key] = data[key][choice]
        
    return data

def split_by_types(data):
    mask = data['types'] == 0
    choosed_train_data_0 = {key:item[mask] for key, item in data.items()}
    mask = data['types'] == 1
    choosed_train_data_1 = {key:item[mask] for key, item in data.items()}
    mask = data['types'] == 2
    choosed_train_data_2 = {key:item[mask] for key, item in data.items()}
    mask = data['types'] == 3
    choosed_train_data_3 = {key:item[mask] for key, item in data.items()}

    return choosed_train_data_0, choosed_train_data_1, choosed_train_data_2, choosed_train_data_3

def split_by_centers(data):
    mask = data['codes'] == 3 # test
    choosed_val_data = {key:item[mask] for key, item in data.items()}
    mask = data['codes'] == 2 # train
    choosed_train_data = {key:item[mask] for key, item in data.items()}

    return choosed_train_data, choosed_val_data

def synthesizing_training_data(train_data):
    mask = data['codes'] == 2 # test
    choosed_val_data = {key:item[mask] for key, item in data.items()}
    mask_2 = data['codes'] == 1 # train
    mask_3 = data['codes'] == 3 # train
    mask = mask_2 + mask_3
    choosed_train_data = {key:item[mask] for key, item in data.items()}

    data['data']=np.append(data['data'], ori_data*(1+(np.random.rand()-0.5)*0.1),axis=0)
    data['labels']=np.append(data['labels'],ori_label,axis=0)
    data['codes']=np.append(data['codes'],ori_code,axis=0)
    data['types']=np.append(data['types'],ori_type,axis=0)

    return synthetic_training_data

def split(data, idx, bins=[0.8, 1.2, 1.6], rate=0.8):
    s = data['data'][:, idx]
    dl = s.__len__()
    index = np.arange(dl)
    land = np.logical_and
    lor = np.logical_or
    l = bins.__len__() - 1
    left = -1000
    train_idx = []
    val_idx = []
    for bin in bins:
        mask = land(s>left, lor(s<bin, s==bin))
        ms = mask.sum()
        train_idx += np.random.choice(index[mask], int(rate*ms), replace=False).tolist()
        left = bin
    mask = s > bins[-1]
    ms = mask.sum()
    train_idx += np.random.choice(index[mask], int(rate*ms), replace=False).tolist()
    val_idx = list(set(index) - set(train_idx))

    train_data = {key:item[train_idx] for key, item in data.items()}
    val_data = {key:item[val_idx] for key, item in data.items()}

    return train_data, val_data

def split_by_label(data, bins=[0, 1], rate=0.8):
    s = data['labels']
    dl = s.__len__()
    index = np.arange(dl)
    land = np.logical_and
    lor = np.logical_or
    l = bins.__len__() - 1
    left = -1000
    train_idx = []
    val_idx = []
    for bin in bins:
        mask = land(s>left, lor(s<bin, s==bin))
        ms = mask.sum()
        train_idx += np.random.choice(index[mask], int(rate*ms), replace=False).tolist()
        left = bin
    mask = s > bins[-1]
    ms = mask.sum()
    train_idx += np.random.choice(index[mask], int(rate*ms), replace=False).tolist()

    val_idx = list(set(index) - set(train_idx))


    train_data = {key:item[train_idx] for key, item in data.items()}
    val_data = {key:item[val_idx] for key, item in data.items()}

    return train_data, val_data

def load_data1_split_idx(split='train'):
    return np.loadtxt('data_split/%s.txt' % split, delimiter='\n', dtype=np.int)

def load_split_idx(split='train'):
    return np.loadtxt('/home/congfang/works/medic/methods/item_analysis/update/data_split/%s.txt' % split, delimiter='\n', dtype=np.int)

ONE_HOT = ['华南市场接触史', '家庭聚集发病', '医务人员接触感染', '是否现吸烟', '高血压', 
'糖尿病', '心梗', '脑梗', '肺结核', '恶性肿瘤', '心脑血管病', '慢性消化系统疾病', 
'外周血管病', '慢性肝或肾功能不全', '慢性呼吸道疾病', '自身免疫或结缔组织病', 
'服用糖皮质激素强的松15mgd×30天以上', '发热', '咳嗽', '喘气', '呼吸急促呼吸困难', 
'发病第几天出现气喘', '咳痰', '乏力', '肌痛', '咯血', '胃肠道症状或吐泻', '头痛', 
'新冠核酸检测_0阴1阳']

def data_normalize(config, data, method='train'):
    if config.normalize == 'mean_shift':

        if method == 'train':     
            config.items_mean = data['data'].mean(axis=0)
        data['data'] -= config.items_mean[None, :]
        # print('Mean shift complete.')

    elif config.normalize == 'minmaxscale':
        if method == 'train':
            config.items_max = data['data'].max(axis=0)
            config.items_min = data['data'].min(axis=0)

        data['data'] -= config.items_min[None, :]
        data['data'] /= (config.items_max[None, :] - config.items_min[None, :] + 1e-8)
        # print('Indicator-wise min-max scaling complete.')

    elif config.normalize == 'mean':
        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.items_min = data['data'].min(axis=0)
            config.items_max = data['data'].max(axis=0)

        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.items_max[None, :] - config.items_min[None, :] + 1e-8)
        # print('Indicator-wise mean scaling complete.')

    elif config.normalize == 'l1':
        if method == 'train':
            config.items_l1_length = np.linalg.norm(data['data'], ord=1, axis=0)

        data['data'] /= (config.items_l1_length + 1e-8)
        # print('Indicator-wise l1 scaling complete.')

    elif config.normalize == 'l2':
        if method == 'train':
            config.items_l2_length = np.linalg.norm(data['data'], ord=2, axis=0)

        data['data'] /= (config.items_l2_length + 1e-8)
        # print('Indicator-wise l2 scaling complete.')

    elif config.normalize == 'Z-Score_indicator-wise':
        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.item_std = data['data'].std(axis=0)
        
        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.item_std[None, :] + 1e-8)
        # print('Indicator-wise Z-Score complete.')

    elif config.normalize == 'qt_z':
        ## === qt transformation ===
        from sklearn.preprocessing import QuantileTransformer
        rng = np.random.RandomState(304)
        if method == 'train':
            qt = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                         random_state=rng)
            qt_transformer = qt.fit(data['data'])
            config['qt_transformer'] = qt_transformer
        data['data'] = config['qt_transformer'].transform(data['data'])
        print('qt transform complete')
        # === z transformation ===
        config.normalize = 'Z-Score_indicator-wise'
        data_normalize(config, data, method)
        config.normalize = 'qt_z'

    elif config.normalize == 'Z-Score_indicator-wise_v2':
        keep_items = config.keep_items.tolist()

        # ======if x is continuous variable, x -> log(1+x)
        idx = [keep_items.index(item) for item in keep_items if item not in ONE_HOT]
        data['data2'] = deepcopy(data['data'])
        data['data3'] = deepcopy(data['data2'])
        data['data2'][:, idx] = np.log(1+data['data2'][:, idx])
        data['data3'][:, idx] = (1+data['data3'][:, idx])**0.5
        # ======

        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.item_std = data['data'].std(axis=0)

            config.items_mean_c2 = data['data2'].mean(axis=0)
            config.item_std_c2 = data['data2'].std(axis=0)

            config.items_mean_c3 = data['data3'].mean(axis=0)
            config.item_std_c3 = data['data3'].std(axis=0)
        
        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.item_std[None, :] + 1e-8)
        
        data['data2'] -= config.items_mean_c2[None, :]
        data['data2'] /= (config.item_std_c2[None, :] + 1e-8)

        data['data3'] -= config.items_mean_c3[None, :]
        data['data3'] /= (config.item_std_c3[None, :] + 1e-8)
        print('Indicator-wise Z-Score complete.')

    elif config.normalize == 'Z-Score_sample-wise':
        data['data'] = preprocessing.scale(data['data'], axis=1)
        print('Sample-wise Z-Score complete.')

    elif config.normalize == 'Z-Score_indicator_sample_wise':
        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.item_std = data['data'].std(axis=0)
            
        data['data2'] = preprocessing.scale(data['data'], axis=1)
        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.item_std[None, :] + 1e-8)

        print('Indicator and sample-wise Z-Score complete.')

    elif config.normalize == 'Z-Score_indicator-wise_l1_sample-wise':
        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.item_std = data['data'].std(axis=0)
        
        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.item_std[None, :] + 1e-8)
        print('Indicator-wise Z-Score complete.')

        # sample-wise l1 norm
        config.samples_l1_length = np.linalg.norm(data['data'], ord=1, axis=1)
        config.samples_l1_length = np.repeat(np.expand_dims(config.samples_l1_length,1),data['data'].shape[1],axis=1)
        data['data'] /= (config.samples_l1_length + 1e-8)
        print('Sample-wise l1 normalization complete.')

    elif config.normalize == 'Z-Score_indicator-wise_l2_sample-wise':
        if method == 'train':
            config.items_mean = data['data'].mean(axis=0)
            config.item_std = data['data'].std(axis=0)
        
        data['data'] -= config.items_mean[None, :]
        data['data'] /= (config.item_std[None, :] + 1e-8)
        print('Indicator-wise Z-Score complete.')

        # sample-wise l2 norm
        config.samples_l2_length = np.linalg.norm(data['data'], ord=2, axis=1)
        config.samples_l2_length = np.repeat(np.expand_dims(config.samples_l2_length,1),data['data'].shape[1],axis=1)
        data['data'] /= (config.samples_l2_length + 1e-8)
        print('Sample-wise l1 normalization complete.')


    elif config.normalize == None: 
        print('With no data normalize.')
    assert np.isnan(data['data']).sum() == 0
    # print('Data loading complete, %d indicators reserved.' % config.mlp_indim)
    return data

def load_data(config, method='test'):
    remove_items = config.remove_items
    with open('%s.pickle.zhuyuan' %config.data, 'rb') as f:
        data = pickle.load(f)
        try:
            items = pickle.load(f)
        except:
            print('items loading is not supported for old version dataset!')
            raise ValueError

        keep = [item not in remove_items for item in items]
        data['labels'] = data['labels'][:, 0]
        data['data'] = data['data'][:, keep]
        data['labels'][data['labels']==2] = 1 if config.num_classes == 2 else 2

        config.keep_items = np.array(items)[keep]
        
    return data