import numpy as np
import h5py
import torch
import glob
import os
from utils import load_json, load_config
from torch.utils.data import Dataset
import pdb

class idrad_tba(Dataset):
    def __init__(self, data_type, config, transform=None, in_memory=False):
        self.data_root = config.data_path
        self.type = data_type
        self._transform = transform
        if self.type == "train":
            self.ann_path = os.path.join(self.data_root, config.train_ann)
            self.data_path = os.path.join(self.data_root, "train")
        elif self.type == "val":
            self.ann_path = os.path.join(self.data_root, config.val_ann)
            self.data_path = os.path.join(self.data_root, "val")
        elif self.type == "test":
            self.ann_path = os.path.join(self.data_root, config.test_ann)
            self.data_path = os.path.join(self.data_root, "test")
        
        self.radar_ann = load_json(self.ann_path) # len(self.radar_ann) = 12078
        self.max_rad_len = config.max_rad_len # 900
        if in_memory:
            self.data = torch.zeros(len(self.radar_ann), self.max_rad_len, 256)
            self.len = torch.ones(len(self.radar_ann))
            for i in range(len(self.radar_ann)):
                k = self.radar_ann[i]['dur']
                self.len[i] = k
                pth = str(self.radar_ann[i]['radar_id']) + '.npy'
                get_data = torch.tensor(np.load(os.path.join(self.data_path, pth)))
                self.data[i][:k] = get_data
        

    def __len__(self):
        return len(self.radar_ann)
        
    def __getitem__(self, index):

        sample = self.data[index]
        sample_len = self.len[index]
        sample_ann = self.radar_ann[index]
        
        if self._transform is not None:
            sample = self._transform(sample)
        
        # labels = torch.stack([data[1]['st'][0],data[1]['st'][1]], dim=1)
        labels = sample_ann['st']
        gap = labels[1] - labels[0] + 1
        # len 900 tensor #  self.max_rad_len
        fl_labels = torch.zeros(self.max_rad_len)
        st_labels = torch.zeros(self.max_rad_len)
        ed_labels = torch.zeros(self.max_rad_len)

        for i in range(labels[0], labels[1] + 1):
            fl_labels[i] = 1
            
        st_labels[labels[0]] = 1
        ed_labels[labels[1]] = 1
        
        sample_ann['fl_labels'] = fl_labels
        sample_ann['st_labels'] = st_labels
        sample_ann['ed_labels'] = ed_labels
        sample_ann['gap'] = gap
        
        return sample, sample_ann
    
    def statistics(self):
        # impossible by the memory issues.
        tmp = []
        avg = self.data.sum()/(self.len.sum() * 256)

        msk = torch.arange(self.max_rad_len,dtype=torch.int16).view(1,-1)
        msk = msk.repeat(self.data.shape[0],1).view(-1, self.max_rad_len, 1)
        msk = msk.repeat(1,1,256)

        cut = self.len.view(-1,1,1)
        cut = cut.to(torch.int16)
        cut = cut.repeat(1, self.max_rad_len, 256)

        msk_a = msk < cut

        z = self.data.to(dtype=torch.float16) - avg.to(dtype=torch.float16)
        val_a = z*z*msk_a
        #val_a = (self.data.to(dtype=torch.float16) - avg.to(dtype=torch.float16))*(self.data.to(dtype=torch.float16) - avg.to(torch.float16))*msk_a
        std = val_a.sum()/(self.len.sum() * 256)
        




class HDF5Dataset(Dataset):

    def __init__(self, folder, labels, feature, random_shift=False, sample_length=150,
                                                shift=15,
                                                transform=None,
                                                in_memory=False):

        self._files = glob.glob(os.path.join(folder, '*.hdf5'))
        self._in_memory = in_memory
        self._sample_length = sample_length
        self._shift = shift
        self._feature = feature
        self._transform = transform
        self._random_shift = random_shift

        self._data = []
        self.frms_per_seq = []
        targets = []
        for index, filename in enumerate(self._files):

            label = filename.split('/')[-1].split('_')[0]
            if label not in labels:
                continue

            with h5py.File(filename, 'r') as file:

                self.frms_per_seq.append(file[self._feature].shape[0])

                targets.append(labels.index(label))
                if self._in_memory:
                    self._data.append(file[self._feature][:])



        self._indices = []
        self._targets = []
        self._saved_data = {}
        for fi in range(len(self.frms_per_seq)):
            for si in range(0, max(1, self.frms_per_seq[fi] - 45), self._shift):
            #for si in range(0, max(1, self.frms_per_seq[fi] - self._sample_length), self._shift):
                self._indices.append((fi, si))
                self._targets.append(targets[fi])

        self._targets = torch.from_numpy(np.asarray(self._targets))
        self._length = len(self._indices)


    def __getitem__(self, index):

        file_index, sequence_index = self._indices[index]

        if self._random_shift:
            sequence_index = np.random.randint(0, self.frms_per_seq[file_index] - self._sample_length)

        if self._in_memory:
            sample = self._data[file_index][sequence_index: sequence_index + self._sample_length]
        else:
            with h5py.File(self._files[file_index], 'r') as file:
                sample = file[self._feature][sequence_index:sequence_index + self._sample_length]

        if len(sample) < self._sample_length:
            sample = np.concatenate((sample, np.repeat(sample[-1][np.newaxis], self._sample_length - len(sample), axis=0)))

        if self._transform is not None:
            sample = self._transform(sample)

        return sample, self._targets[index]

    def __len__(self):
        return self._length

if __name__ == '__main__':
    config = load_config('./config/data_config.json')
    data = idrad_tba("train", config, in_memory=True)
    data.statistics()
    pdb.set_trace()
