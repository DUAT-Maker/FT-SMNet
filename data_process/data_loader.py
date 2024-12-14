import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 data_path='FuXing.csv', scale=True, inverse=False, empty_ratio=0):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.empty_ratio = empty_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        df_stamp = pd.to_datetime(df_raw['Time'])
        df_raw.drop(labels='Time', axis = 1, inplace=True)
        num_train = int(len(df_raw)*0.6)
        num_test = int(len(df_raw)*0.3)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]        
        cols_data = df_raw.columns
        df_data = df_raw[cols_data]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values  
        self.data_x = data[border1:border2]     
        df_stamp = df_stamp.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df_stamp = df_stamp.apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.data_stamp = df_stamp.values[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.empty_ratio:
            nums = self.seq_len * 9
            arr = np.random.choice(nums, nums, replace=False)
            arr = arr.reshape(self.seq_len, 9)
            arr_0 = arr > int(nums*self.empty_ratio)
            seq_x = arr_0 * seq_x
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark , seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



