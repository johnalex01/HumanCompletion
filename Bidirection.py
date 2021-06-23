import torch.nn as nn
from Attention import AttentionModel,AttentionModel_without_Residual,AttentionModel_for_single_side_attention
import torch
from Models import GC_Block, PostGCN
from torch.nn.parameter import Parameter

class MergeNet(nn.Module):
    def __init__(self, seq_len, pose_dim):
        super(MergeNet, self).__init__()

        self.block1 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.block2 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.block3 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.post_net = PostGCN(pose_dim, 66, node_n=seq_len)

        #self.lin1 = nn.Linear(pose_dim, 256)
        self.lin1 = nn.Linear(66, 66)
        self.dp1 = nn.Dropout(0.3)
        self.act1 = nn.LeakyReLU(0.2)

        # self.lin2 = nn.Linear(256, 66)
        # self.dp2 = nn.Dropout(0.3)
        # self.act2 = nn.LeakyReLU(0.2)


        self.W1 = Parameter(torch.FloatTensor(25, 1))
        self.W2 = Parameter(torch.FloatTensor(25, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.W1.data.fill_(0.5);
        self.W2.data.fill_(0.5);


    def forward(self, x, y, for_resid):
        # x = torch.mul(x,self.W1)
        # y = torch.mul(y,self.W2)
        xy = torch.add(x, y)
        xy = xy/2

        #xy = torch.cat((x, y), dim=-1)
        # xy = self.lin1(xy)
        # xy = self.dp1(xy)
        #xy = self.act1(xy)
        # xy = self.lin2(xy)
        # xy = self.dp2(xy)
        # xy = self.act2(xy)

        # xy = torch.cat((x, y), dim=-1)
        # xy = self.block1(xy)
        # xy = self.block2(xy)
        # xy = self.block3(xy)
        # xy = self.post_net(xy)

        #xy = xy + for_resid

        return xy


class MergeNet_without_Residual(nn.Module):
    def __init__(self, seq_len, pose_dim):
        super(MergeNet_without_Residual, self).__init__()

        self.block1 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.block2 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.block3 = GC_Block(pose_dim, p_dropout=0.3, bias=True, node_n=seq_len)
        self.post_net = PostGCN(pose_dim, 66, node_n=seq_len)

    def forward(self, x, y, for_resid):
        # xy = torch.add(x, y)
        # xy = xy/2
        xy = torch.cat((x, y), dim=-1)
        xy = self.block1(xy)
        xy = self.block2(xy)
        xy = self.block3(xy)
        xy = self.post_net(xy)

       # xy = xy + for_resid

        return xy



class BidirectionModel(nn.Module):
    def __init__(self):
        super(BidirectionModel, self).__init__()
        self.model = AttentionModel()
        self.inverse_model = AttentionModel()
        self.merge = MergeNet(25, 66 * 2)

    def forward(self, x, z, for_resid, invx, invz, inv_for_resid):
        out = self.model(x, z, for_resid)
        inv_out = self.inverse_model(invz, invx, inv_for_resid)
        inv_inv_out = inv_out.flip(1)

        ret = self.merge(out, inv_inv_out, for_resid)

        return out, inv_out, ret


class BidirectionModel_for_single_side_attention(nn.Module):
    def __init__(self):
        super(BidirectionModel_for_single_side_attention, self).__init__()
        self.model = AttentionModel_for_single_side_attention()
        self.inverse_model = AttentionModel_for_single_side_attention()
        self.merge = MergeNet(25, 66 * 2)

    def forward(self, x, z, for_resid, invx, invz, inv_for_resid):
        out = self.model(x, z, for_resid)
        inv_out = self.inverse_model(invz, invx, inv_for_resid)
        inv_inv_out = inv_out.flip(1)

        ret = self.merge(out, inv_inv_out, for_resid)

        return out, inv_out, ret



class BidirectionModel_without_Residual(nn.Module):
    def __init__(self):
        super(BidirectionModel_without_Residual, self).__init__()
        self.model = AttentionModel_without_Residual()
        self.inverse_model = AttentionModel_without_Residual()
        self.merge = MergeNet_without_Residual(25, 66 * 2)

    def forward(self, x, z, for_resid, invx, invz):
        out = self.model(x, z,)
        inv_out = self.inverse_model(invz, invx)
        inv_inv_out = inv_out.flip(1)
        ret = self.merge(out, inv_inv_out, for_resid)

        return out, inv_out, ret

