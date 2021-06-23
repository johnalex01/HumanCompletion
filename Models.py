import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math


class ToPose(nn.Module):
    def __init__(self, seq_len, lstm_hidden_size):
        super(ToPose, self).__init__()

        self.lin1 = nn.Linear(lstm_hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(seq_len * 512)
        # self.dp1 = nn.Dropout(0.3)
        self.act1 = nn.LeakyReLU(0.2)

        self.lin2 = nn.Linear(512, 66)
        # self.act2 = nn.Tanh()

    def forward(self, x):
        y = self.lin1(x)
        b, s, d = y.shape
        y = self.bn1(y.view(b, s * d).contiguous()).view(b, s, d).contiguous()
        y = self.act1(y)
        # y = self.dp1(y)

        y = self.lin2(y)
        # y = self.act2(y)

        return y


class ToFeature(nn.Module):
    def __init__(self, seq_len, pose_dim):
        super(ToFeature, self).__init__()

        self.lin1 = nn.Linear(pose_dim, 256)
        self.bn1 = nn.BatchNorm1d(seq_len * 256)
        self.dp1 = nn.Dropout(0.3)
        self.act1 = nn.LeakyReLU(0.2)

        self.lin2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(seq_len * 512)
        self.dp2 = nn.Dropout(0.3)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.lin1(x)
        b, s, d = y.shape
        y = self.bn1(y.view(b, s * d).contiguous()).view(b, s, d).contiguous()
        y = self.act1(y)
        y = self.dp1(y)

        y = self.lin2(y)
        b, s, d = y.shape
        y = self.bn2(y.view(b, s * d).contiguous()).view(b, s, d).contiguous()
        y = self.act2(y)
        y = self.dp2(y)

        return y


class AttentionLayer(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(AttentionLayer, self).__init__()
        self.W = Parameter(torch.FloatTensor(lstm_hidden_size, lstm_hidden_size))
        self.V = Parameter(torch.FloatTensor(lstm_hidden_size, lstm_hidden_size))
        self.bias = Parameter(torch.FloatTensor(21, lstm_hidden_size))
        self.Wt = Parameter(torch.FloatTensor(lstm_hidden_size, 1))
        self.tan = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.Wt.data.uniform_(-stdv, stdv)

    def forward(self, c, allh):
        #print(['c:', c.shape])
        c = c.permute(1, 0, 2).contiguous()
        #print(['c_permute:', c.shape])
        # print(['c_repeat:', c.shape])
        allh = allh.permute(1, 0, 2).contiguous()
        # print(['allh:', allh.shape])

        wc = torch.matmul(c, self.W)
        wc = wc.repeat(1, allh.shape[1], 1)

        vh = torch.matmul(allh, self.V)

        # print(['wc:', wc.shape])
        # print(['vh:', vh.shape])
        output = wc + vh + self.bias
        # print(['output:', output.shape])
        output = self.tan(output)
        # print(['output:', output.shape])
        output = torch.matmul(output, self.Wt)
        # print(['output:', output.shape])
        alpha = self.softmax(output)
        # print(['alpha:', alpha.shape])
        y = alpha * allh
        # print(['y:', y.shape])
        y = torch.sum(y, dim=1, keepdim=True)
        # print(['y:', y.shape])
        y = y.permute(1, 0, 2).contiguous()
        # print(['y:', y.shape])
        return y

class AttentionLayer_for_single_side_attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(AttentionLayer_for_single_side_attention, self).__init__()
        self.W = Parameter(torch.FloatTensor(lstm_hidden_size, lstm_hidden_size))
        self.V = Parameter(torch.FloatTensor(lstm_hidden_size, lstm_hidden_size))
        self.bias = Parameter(torch.FloatTensor(10, lstm_hidden_size))
        self.Wt = Parameter(torch.FloatTensor(lstm_hidden_size, 1))
        self.tan = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.Wt.data.uniform_(-stdv, stdv)

    def forward(self, c, allh):
        # print(['c:', c.shape])
        c = c.permute(1, 0, 2).contiguous()
        # print(['c_permute:', c.shape])
        # print(['c_repeat:', c.shape])
        allh = allh.permute(1, 0, 2).contiguous()
        # print(['allh:', allh.shape])

        wc = torch.matmul(c, self.W)
        wc = wc.repeat(1, allh.shape[1], 1)

        vh = torch.matmul(allh, self.V)

        # print(['wc:', wc.shape])
        # print(['vh:', vh.shape])
        output = wc + vh + self.bias
        # print(['output:', output.shape])
        output = self.tan(output)
        # print(['output:', output.shape])
        output = torch.matmul(output, self.Wt)
        # print(['output:', output.shape])
        alpha = self.softmax(output)
        # print(['alpha:', alpha.shape])
        y = alpha * allh
        # print(['y:', y.shape])
        y = torch.sum(y, dim=1, keepdim=True)
        # print(['y:', y.shape])
        y = y.permute(1, 0, 2).contiguous()
        # print(['y:', y.shape])
        return y

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = y + x
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PostGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n):
        super(PostGCN, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n

        self.gcn = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        # self.act_f = nn.Sigmoid()
        # self.act_f = nn.LeakyReLU(option.leaky_c)  # 最后一层加激活不确定对不对
        # self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gcn(x)
        # y = self.act_f(y)  # 最后一层加激活不确定对不对
        return y

