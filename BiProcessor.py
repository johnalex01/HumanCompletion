from Attention import AttentionModel
from Seq2seq import Seq2Seq
from Bidirection import BidirectionModel
from DatasetTrain import DatasetTrain
from DatasetTest import DatasetTest
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from Tools import action_keys, dim_used_3d, dim_repeat_22, dim_repeat_32, denorm, L2NormLoss_train, L2NormLoss_test


dataString = "ndataset"
logString = '10_25_10'

class BiProcessor(object):
    def __init__(self):
        self.model = BidirectionModel()
        self.model.cuda()  # 这行已经放在 显卡了
        self.logfile = open('_log/logfile_'+logString+'.txt','a+')
        print("new workers:",file=self.logfile)
        print(">>> Total params: {:.2f}M\n".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        #print(self.model.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.train_dataset = DatasetTrain('_dataset/'+dataString+'/npy_data/total_train_ab.npy', mode_name='train')
        self.train_dataset_loader = DataLoader(self.train_dataset, shuffle=True, num_workers=2, batch_size=16)

        self.all_test_datasets = []
        self.all_test_dataset_loaders = []

        self.all_test_datasets_32 = []
        self.all_test_dataset_loaders_32 = []

        for key in action_keys:
            testd = DatasetTest('_dataset/'+dataString+'/test/test_npy/', key, '_22.npy',
                                self.train_dataset.global_max, self.train_dataset.global_min)
            testd_loader = DataLoader(testd, shuffle=False, num_workers=0, batch_size=128)
            self.all_test_datasets.append(testd)
            self.all_test_dataset_loaders.append(testd_loader)

            testd32 = DatasetTest('_dataset/'+dataString+'/test/test_npy32/', key, '_32.npy',
                                  self.train_dataset.global_max,
                                  self.train_dataset.global_min)
            testd_loader32 = DataLoader(testd32, shuffle=False, num_workers=0, batch_size=128)
            self.all_test_datasets_32.append(testd32)
            self.all_test_dataset_loaders_32.append(testd_loader32)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        print(epoch)
        print(epoch,file=self.logfile)
        #self.logfile.flush()
        for i, data in enumerate(self.train_dataset_loader, 0):
            data_pre = data['pre_10'].float().cuda()
            data_post = data['post_10'].float().cuda()
            data_gt = data['mid_25'].float().cuda()
            data_resid = data['mid_resid'].float().cuda()

            inv_data_pre = data_pre.flip(1)
            inv_data_post = data_post.flip(1)
            inv_data_gt = data_gt.flip(1)
            inv_data_resid = data_resid.flip(1)

            out, inv_out, ret = self.model(data_pre, data_post, data_resid, inv_data_post, inv_data_pre, inv_data_resid)

            data_gt = denorm(data_gt, self.train_dataset.global_max, self.train_dataset.global_min)
            out = denorm(out, self.train_dataset.global_max, self.train_dataset.global_min)

            inv_data_gt = denorm(inv_data_gt, self.train_dataset.global_max, self.train_dataset.global_min)
            inv_out = denorm(inv_out, self.train_dataset.global_max, self.train_dataset.global_min)

            ret = denorm(ret, self.train_dataset.global_max, self.train_dataset.global_min)

            loss1 = L2NormLoss_train(data_gt, out)
            loss2 = L2NormLoss_train(inv_data_gt, inv_out)
            loss3 = L2NormLoss_train(data_gt, ret)

            loss =0.5* loss1 +0.5*  loss2 + 2* loss3

            running_loss = running_loss + loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 150 == 0 and i > 0:
                print(['running loss: %f' % (running_loss / 150)])
                print(['running loss: %f' % (running_loss / 150)],file=self.logfile)
               # self.logfile.flush()
                running_loss = 0

    def test(self):
        self.model.eval()

        frame_ids = [1, 3, 7, 9, 13, 24]
        # frame_ids = range(1, 25)
        total_loss = np.zeros((len(action_keys), len(frame_ids)))
        total_loss_32 = np.zeros((len(action_keys), len(frame_ids)))

        for i in range(len(self.all_test_datasets)):
            count = 0
            #total_loss = np.zeros((len(action_keys), len(frame_ids)))
            #total_loss_32 = np.zeros((len(action_keys), len(frame_ids)))

            for data22, data32 in zip(self.all_test_dataset_loaders[i], self.all_test_dataset_loaders_32[i]):
                input_pre_22 = data22['pre_10'].float().cuda()
                input_post_22 = data22['post_10'].float().cuda()
                input_mid_resid = data22['mid_resid'].float().cuda()

                inv_pre_22 = input_pre_22.flip(1)
                inv_post_22 = input_post_22.flip(1)
                inv_mid_resid = input_mid_resid.flip(1)

                gt22 = data22['mid_25'].float().cuda()
                groundtruth32 = data32['mid_25'].float().cuda()

                with torch.no_grad():
                    _, _, outputs = self.model(input_pre_22, input_post_22, input_mid_resid, inv_post_22, inv_pre_22,
                                               inv_mid_resid)

                    mygt = groundtruth32.view(-1, 25, 32, 3).contiguous().clone()
                    myout = outputs.view(-1, 25, 22, 3).contiguous()
                    mygt[:, :, dim_used_3d, :] = myout
                    mygt[:, :, dim_repeat_32, :] = myout[:, :, dim_repeat_22, :]
                    mygt = mygt.view(-1, 25, 96).contiguous()

                    groundtruth32 = denorm(groundtruth32, self.train_dataset.global_max, self.train_dataset.global_min)
                    mygt = denorm(mygt, self.train_dataset.global_max, self.train_dataset.global_min)
                    loss32 = L2NormLoss_test(groundtruth32, mygt, frame_ids).cpu().data.numpy()

                    outputs = denorm(outputs, self.train_dataset.global_max, self.train_dataset.global_min)
                    gt22 = denorm(gt22, self.train_dataset.global_max, self.train_dataset.global_min)
                    loss = L2NormLoss_test(gt22, outputs, frame_ids).cpu().data.numpy()

                    total_loss[i] += loss
                    total_loss_32[i] += loss32
                    count += 1
            total_loss[i] /= count
            total_loss_32[i] /= count
            # print(count)

        # print(total_loss)
        # print(np.mean(total_loss, axis=0))
        print(['l2 loss of 22 points:', np.mean(total_loss)])

        print(total_loss_32)
        print(np.mean(total_loss_32, axis=0))
        print(['l1 loss of 22 points:', np.mean(total_loss_32)])

        print(['l2 loss of 22 points:', np.mean(total_loss)],file=self.logfile)

        print(total_loss_32,file=self.logfile)
        print(np.mean(total_loss_32, axis=0),file=self.logfile)
        print(['l1 loss of 22 points:', np.mean(total_loss_32)],file=self.logfile)

        return np.mean(total_loss_32)

    def train_and_test(self):
        self.l1_loss= 1000.0
        for epoch in range(100):
            self.train_one_epoch(epoch)
            new_l1_loss=self.test()
            if(new_l1_loss<self.l1_loss):
                torch.save(self.model,logString+'.pth')
                self.l1_loss=new_l1_loss
            self.logfile.flush()
        self.logfile.close()
