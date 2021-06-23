from torch.utils.data import Dataset
import numpy as np
import os


class DatasetTest(Dataset):
    def __init__(self, dataset_path, action_type, postex, global_max, global_min):
        super(DatasetTest, self).__init__()

        self.dataset_path = dataset_path + action_type + postex

        self.seq_45 = np.load(self.dataset_path)
        self.seq_45 = np.transpose(self.seq_45, (0, 2, 1))

        

        self.global_max = global_max
        self.global_min = global_min

        # Normalize
        self.seq_45 = (self.seq_45 - self.global_min) / (self.global_max - self.global_min)
        self.seq_45 = self.seq_45 * 2 - 1

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

        #15_25_10
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

        # print(self.seq_45.shape)
        self.seq_pre_10 = self.seq_45[:, 10:20, :]
        self.seq_mid_25 = self.seq_45[:, 20:45, :]
        self.seq_post_10 = self.seq_45[:, 45:55, :]

        last_x = self.seq_45[:, 19, :]
        first_z = self.seq_45[:, 45, :]


        self.mid_resid = np.linspace(last_x, first_z, 25).transpose((1, 0, 2))

        # self.mid_repx = np.repeat(self.seq_45[:, 9:10, :], (1, 25, 1))
        # self.mid_repz = np.repeat(self.seq_45[:, 35:36, :], (1, 25, 1))

        # self.seq_pre_10 = self.seq_pre_10[:48]
        # self.seq_mid_25 = self.seq_mid_25[:48]
        # self.mid_resid = self.mid_resid[:48]
        # self.seq_post_10 = self.seq_post_10[:48]

        self.data_num = len(self.seq_pre_10)

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        output = {
            'pre_10': self.seq_pre_10[item],
            'mid_25': self.seq_mid_25[item],
            'mid_resid': self.mid_resid[item],
            'post_10': self.seq_post_10[item],
            # 'mid_repx':self.mid_repx[item],
            # 'mid_repz': self.mid_repz[item]
        }
        return output
