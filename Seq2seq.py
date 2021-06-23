import torch
from Models import ToPose, ToFeature
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.input_seq_dim = (10, 66)
        self.output_seq_dim = (25, 66)
        self.hidden_size = 1024

        self.to_feature = ToFeature(seq_len=self.input_seq_dim[0], pose_dim=self.input_seq_dim[1])
        self.encoder = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.decoder = nn.LSTM(input_size=self.output_seq_dim[1], hidden_size=self.hidden_size)
        self.to_pose = ToPose(seq_len=self.output_seq_dim[0], lstm_hidden_size=self.hidden_size)

    def forward(self, x):
        x = self.to_feature(x)

        b, _, _ = x.shape

        x = x.permute(1, 0, 2).contiguous()

        _, (h_n, c_n) = self.encoder(x)

        zero_mid = torch.zeros((self.output_seq_dim[0], b, self.output_seq_dim[1])).cuda()
        # zero_cn = torch.zeros_like(c_n).cuda()
        # zero_hn = torch.zeros_like(h_n).cuda()

        o, (_, _) = self.decoder(zero_mid, (h_n, c_n))

        o = o.permute(1, 0, 2).contiguous()

        o = self.to_pose(o)

        return o



class BiSeq2Seq(nn.Module):
    def __init__(self):
        super(BiSeq2Seq, self).__init__()
        self.input_seq_dim = (20, 66)
        self.output_seq_dim = (25, 66)
        self.hidden_size = 1024

        self.to_feature = ToFeature(seq_len=self.input_seq_dim[0], pose_dim=self.input_seq_dim[1])
        self.encoder = nn.LSTM(input_size=512, hidden_size=self.hidden_size,bidirectional=True)
        self.decoder = nn.LSTM(input_size=self.output_seq_dim[1], hidden_size=self.hidden_size,bidirectional=True)
        self.to_pose = ToPose(seq_len=self.output_seq_dim[0], lstm_hidden_size=self.hidden_size*2)

    def forward(self, x):
        x = self.to_feature(x)

        b, _, _ = x.shape

        x = x.permute(1, 0, 2).contiguous()

        _, (h_n, c_n) = self.encoder(x)
        # print(h_n.shape)

        zero_mid = torch.zeros((self.output_seq_dim[0], b, self.output_seq_dim[1])).cuda()
        # zero_cn = torch.zeros_like(c_n).cuda()
        # zero_hn = torch.zeros_like(h_n).cuda()

        o, (_, _) = self.decoder(zero_mid, (h_n, c_n))
        #print(o.shape)

        o = o.permute(1, 0, 2).contiguous()

        o = self.to_pose(o)

        return o
