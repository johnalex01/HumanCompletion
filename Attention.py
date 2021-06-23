import torch.nn as nn
from Models import ToFeature, ToPose, AttentionLayer,AttentionLayer_for_single_side_attention
import torch


class DecoderModel(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(DecoderModel, self).__init__()

        self.cell = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size)
        self.to_feature = ToFeature(seq_len=1, pose_dim=66)
        self.to_pose = ToPose(seq_len=1, lstm_hidden_size=lstm_hidden_size)
        self.attention_layer = AttentionLayer(lstm_hidden_size=lstm_hidden_size)
        self.lin = nn.Linear(1024, 512)

    def forward(self, all_h, all_hz, h_n, c_n):
        cat_h = torch.cat((all_h, h_n, all_hz), dim=0)
        atth = self.attention_layer(c_n, cat_h)
        # atth = h_n
        # x = self.to_feature(pose)
        # x = x.permute(1, 0, 2).contiguous()
        zero_mid = self.lin(h_n)  # torch.zeros((1, h_n.shape[1], 512)).cuda()
        # zero_x = torch.zeros.cuda()
        #print(['atth:', atth.shape])
        # print(['c_n:', c_n.shape])
        # print(['x:', x.shape])
        _, (new_h, new_c) = self.cell(zero_mid, (atth, c_n))

        # new_pose = self.to_pose(new_h.clone().permute(1, 0, 2).contiguous())

        return new_h, new_c  # , new_pose

class DecoderModel_for_single_side_attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(DecoderModel_for_single_side_attention, self).__init__()

        self.cell = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size)
        self.to_feature = ToFeature(seq_len=1, pose_dim=66)
        self.to_pose = ToPose(seq_len=1, lstm_hidden_size=lstm_hidden_size)
        self.attention_layer = AttentionLayer_for_single_side_attention(lstm_hidden_size=lstm_hidden_size)
        self.lin = nn.Linear(1024, 512)

    def forward(self, all_h, h_n, c_n):
        #cat_h = torch.cat((all_h), dim=0)
        atth = self.attention_layer(c_n, all_h)
        # atth = h_n
        # x = self.to_feature(pose)
        # x = x.permute(1, 0, 2).contiguous()
        zero_mid = self.lin(h_n)  # torch.zeros((1, h_n.shape[1], 512)).cuda()
        # zero_x = torch.zeros.cuda()
        # print(['atth:', atth.shape])
        # print(['c_n:', c_n.shape])
        # print(['x:', x.shape])
        _, (new_h, new_c) = self.cell(zero_mid, (atth, c_n))

        # new_pose = self.to_pose(new_h.clone().permute(1, 0, 2).contiguous())

        return new_h, new_c  # , new_pose

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        # self.input_seq_dim = (10, 66)
        self.input_seq_dim = (10, 66)
        self.output_seq_dim = (25, 66)
        self.hidden_size = 1024

        self.to_feature = ToFeature(seq_len=self.input_seq_dim[0], pose_dim=self.input_seq_dim[1])

        self.encoder = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.encoder_post = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.decoder = DecoderModel(lstm_hidden_size=self.hidden_size)
        # self.decoder = nn.LSTM(input_size=66, hidden_size=self.hidden_size)

        self.to_pose = ToPose(seq_len=25, lstm_hidden_size=self.hidden_size)

    def forward(self, x, z, for_resid):
        output = []

        x = self.to_feature(x)
        z = self.to_feature(z)

        x = x.permute(1, 0, 2).contiguous()
        z = z.permute(1, 0, 2).contiguous()

        all_h, (h_n, c_n) = self.encoder(x)
        #print(['c_n:', c_n.shape])
        all_hz, (_, _) = self.encoder_post(z)

        # new_h = h_n.clone()
        # new_c = c_n.clone()

        for i in range(25):
            h_n, c_n = self.decoder(all_h, all_hz, h_n, c_n)
            output.append(h_n)

        output = torch.cat(output, dim=0).permute(1, 0, 2).contiguous()
        o = self.to_pose(output)

        return o + for_resid


class AttentionModel_without_Residual(nn.Module):
    def __init__(self):
        super(AttentionModel_without_Residual, self).__init__()
        self.input_seq_dim = (10, 66)
        self.output_seq_dim = (25, 66)
        self.hidden_size = 1024

        self.to_feature = ToFeature(seq_len=self.input_seq_dim[0], pose_dim=self.input_seq_dim[1])

        self.encoder = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.encoder_post = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.decoder = DecoderModel(lstm_hidden_size=self.hidden_size)
        # self.decoder = nn.LSTM(input_size=66, hidden_size=self.hidden_size)

        self.to_pose = ToPose(seq_len=25, lstm_hidden_size=self.hidden_size)

    def forward(self, x, z):
        output = []

        x = self.to_feature(x)
        z = self.to_feature(z)

        x = x.permute(1, 0, 2).contiguous()
        z = z.permute(1, 0, 2).contiguous()

        all_h, (h_n, c_n) = self.encoder(x)
        all_hz, (_, _) = self.encoder_post(z)

        # new_h = h_n.clone()
        # new_c = c_n.clone()

        for i in range(25):
            h_n, c_n = self.decoder(all_h, all_hz, h_n, c_n)
            output.append(h_n)

        output = torch.cat(output, dim=0).permute(1, 0, 2).contiguous()
        o = self.to_pose(output)

        return o

class AttentionModel_for_single_side_attention(nn.Module):
    def __init__(self):
        super(AttentionModel_for_single_side_attention, self).__init__()
        self.input_seq_dim = (10, 66)
        self.output_seq_dim = (25, 66)
        self.hidden_size = 1024

        self.to_feature = ToFeature(seq_len=self.input_seq_dim[0], pose_dim=self.input_seq_dim[1])

        self.encoder = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.encoder_post = nn.LSTM(input_size=512, hidden_size=self.hidden_size)
        self.decoder = DecoderModel_for_single_side_attention(lstm_hidden_size=self.hidden_size)
        # self.decoder = nn.LSTM(input_size=66, hidden_size=self.hidden_size)

        self.to_pose = ToPose(seq_len=25, lstm_hidden_size=self.hidden_size)

    def forward(self, x, z, for_resid):
        output = []

        x = self.to_feature(x)
        z = self.to_feature(z)

        x = x.permute(1, 0, 2).contiguous()
        z = z.permute(1, 0, 2).contiguous()

        all_h, (h_n, c_n) = self.encoder(x)

        # new_h = h_n.clone()
        # new_c = c_n.clone()

        for i in range(25):
            h_n, c_n = self.decoder(all_h, h_n, c_n)
            output.append(h_n)

        output = torch.cat(output, dim=0).permute(1, 0, 2).contiguous()
        o = self.to_pose(output)

        return o + for_resid