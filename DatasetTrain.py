from torch.utils.data import Dataset
import numpy as np
import os


class DatasetTrain(Dataset):
    def __init__(self, dataset_path, mode_name, global_max=1e20, global_min=-1e20):
        super(DatasetTrain, self).__init__()

        self.dataset_path = dataset_path

        self.seq_45 = np.load(dataset_path)

        

        if mode_name == 'train':
            self.global_max = np.max(self.seq_45)
            self.global_min = np.min(self.seq_45)
        else:
            self.global_max = global_max
            self.global_min = global_min

        # Normalize
        self.seq_45 = (self.seq_45 - self.global_min) / (self.global_max - self.global_min)
        self.seq_45 = self.seq_45 * 2 - 1

        ma = np.max(self.seq_45)
        mb = np.min(self.seq_45)

        # self.seq_pre_10 = self.seq_45[:, :10, :]
        # self.seq_mid_25 = self.seq_45[:, 10:35, :]
        # self.seq_post_10 = self.seq_45[:, 35:, :]

        # self.seq_pre_10 = self.seq_45[:, 9:10, :]
        # self.seq_mid_25 = self.seq_45[:, 10:35, :]
        # self.seq_post_10 = self.seq_45[:, 35:36, :]

        # self.seq_pre_10 = self.seq_45[:, 5:10, :]
        # self.seq_mid_25 = self.seq_45[:, 10:35, :]
        # self.seq_post_10 = self.seq_45[:, 35:40, :]

        # last_x = self.seq_45[:, 9, :]
        # first_z = self.seq_45[:, 35, :]

        #5_25_5
        # self.seq_pre_10 = self.seq_45[:, :5, :]
        # self.seq_mid_25 = self.seq_45[:, 5:30, :]
        # self.seq_post_10 = self.seq_45[:, 30:, :]

        # last_x = self.seq_45[:, 4, :]
        # first_z = self.seq_45[:, 30, :]

        #10_25_10
        # self.seq_pre_10 = self.seq_45[:, :10, :]
        # self.seq_mid_25 = self.seq_45[:, 10:35, :]
        # self.seq_post_10 = self.seq_45[:, 35:, :]

        # last_x = self.seq_45[:, 9, :]
        # first_z = self.seq_45[:, 35, :]

        #15_25_15
        # self.seq_pre_10 = self.seq_45[:, :15, :]
        # self.seq_mid_25 = self.seq_45[:, 15:40, :]
        # self.seq_post_10 = self.seq_45[:, 40:, :]


        # last_x = self.seq_45[:, 14, :]
        # first_z = self.seq_45[:, 40, :]

        #20_25_20
        # self.seq_pre_10 = self.seq_45[:, :20, :]
        # self.seq_mid_25 = self.seq_45[:, 20:45, :]
        # self.seq_post_10 = self.seq_45[:, 45:, :]

        # last_x = self.seq_45[:, 19, :]
        # first_z = self.seq_45[:, 45, :]


        self.seq_pre_10 = self.seq_45[:, 10:20, :]
        self.seq_mid_25 = self.seq_45[:, 20:45, :]
        self.seq_post_10 = self.seq_45[:, 45:55, :]

        last_x = self.seq_45[:, 19, :]
        first_z = self.seq_45[:, 45, :]

        self.mid_resid = np.linspace(last_x, first_z, 25).transpose((1, 0, 2))

        # self.mid_resid = np.repeat(self.seq_45[:, 9:10, :], (1, 25, 1))

        self.data_num = len(self.seq_post_10)

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        output = {
            'pre_10': self.seq_pre_10[item],
            'mid_25': self.seq_mid_25[item],
            'mid_resid': self.mid_resid[item],
            'post_10': self.seq_post_10[item]
        }
        return output
