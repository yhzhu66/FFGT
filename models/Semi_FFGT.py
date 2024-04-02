import time
from models.embedder import embedder_single
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import copy

def get_A_r(adj, r):
    adj_label = adj
    if r <= 1:
        pass
    else:
        for i in range(r - 1):
            adj_label = adj_label @ adj
    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(x_dis / tau)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def Ncontrast_v1(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(x_dis / tau)
    x_dis_sum = torch.sum(x_dis, 1)

    median_values, _ = torch.median(x_dis, dim=1, keepdim=True)
    x_dis_bi = torch.where(x_dis > median_values, x_dis, 0.)
    x_dis_sum_pos = torch.sum(x_dis_bi*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss


def Contrast_hl(anchor, positive, negative,margin = 1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)

    loss = torch.mean(F.relu(distance_positive - distance_negative + margin))
    return loss

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 0) #-9e15

    scores_full = F.softmax(scores, dim=-1)
    scores_full = dropout(scores_full)
    output_full = torch.matmul(scores_full, v)
    return output_full

class MultiHeadAttention_new(nn.Module):
    def __init__(self, args, d_model_in, d_model_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model_out
        self.d_k = d_model_out
        self.q_linear = nn.Linear(d_model_in, d_model_out)
        self.v_linear = nn.Linear(d_model_in, d_model_out)
        self.k_linear = nn.Linear(d_model_in, d_model_out)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model_out, d_model_out)

        self.lamb = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, q, k, v, mask=None):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        return scores


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class FeedForward_first(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_ff)

    def forward(self, x):
        x1 = self.dropout(F.relu(self.linear_1(x)))
        x2 = self.linear_2(x1)
        return x1, x2

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, args, d_model, dropout=0.1):
        super().__init__()
        self.args = args
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        # self.effectattn = EfficientAttention(in_channels = d_model, key_channels =d_model, head_count =heads, value_channels = d_model)
        self.MHattn = MultiHeadAttention_new(args=args, d_model_in = d_model, d_model_out = d_model,dropout = self.args.dropout_att)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lamb = nn.Parameter(torch.zeros(2), requires_grad=True)
        # self.linear_dim = nn.Linear(3 * d_model, d_model)

    def forward(self, x, adj = None):
        x = self.norm_1(x)
        att = self.MHattn(x, x, x, adj)
        x = x + att
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x

class FFGT_model(nn.Module):
    def __init__(self, arg):
        super(FFGT_model, self).__init__()
        in_channels = arg.ft_size
        self.arg= arg
        self.nclass = arg.nb_classes
        self.hid_dim = arg.hid_dim
        self.dropout = arg.dropout
        self.Trans_layer_num = arg.Trans_layer_num
        self.norm_layer_input = Norm(in_channels)
        self.norm_layer_mid = Norm(arg.hid_dim)
        self.layers_low = get_clones(EncoderLayer(arg, arg.hid_dim, arg.dropout_att), arg.Trans_layer_num)
        self.layers1_hybrid = get_clones(EncoderLayer(arg, arg.hid_dim, arg.dropout_att), arg.Trans_layer_num)
        self.layers2_high = get_clones(EncoderLayer(arg, arg.hid_dim, arg.dropout_att), arg.Trans_layer_num)
        self.MLP1 = nn.Linear(in_channels, self.hid_dim)
        self.MLP2 = nn.Linear(in_channels, self.hid_dim)
        self.layer_singOUT1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.layer_singOUT2 = nn.Linear(self.hid_dim, self.nclass)
        self.lamb = nn.Parameter(torch.zeros(2), requires_grad=True)


    def forward(self, x_input, adj):
        x_input = self.norm_layer_input(x_input)
        X1 = self.MLP1(x_input)
        x_dis = get_feature_dis(self.norm_layer_mid(X1))
        X_l = self.MLP2(x_input)
        X_b, X_h = X1, X1
        for i in range(self.Trans_layer_num):
            X_l = self.layers_low[i](X_l, adj)
            X_b = self.layers1_hybrid[i](X_b)
            X_h = self.layers2_high[i](X_h)
        C_loss= Contrast_hl(X_h,X_b,X_l)
        X_out = X_b + X_l * self.lamb[0] + X_h * (1. + self.lamb[1])

        X_out = F.elu(self.layer_singOUT1(X_out))
        CONN_INDEX = self.layer_singOUT2(X_out)

        return F.log_softmax(CONN_INDEX, dim=1), x_dis, C_loss

class FFGT(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.args.nb_classes = nb_classes
        self.args.n_sample = self.labels.size(0)
        self.model = FFGT_model(self.args).to(self.args.device)

    def training(self):
        features = self.features.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
        if self.idx_train.dtype == torch.bool:
            self.idx_train = torch.where(self.idx_train == 1)[0]
            self.idx_val = torch.where(self.idx_val == 1)[0]
            self.idx_test = torch.where(self.idx_test == 1)[0]

        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0
        start = time.time()
        totalL = []
        index1 = 0

        adj_label_list = []

        for i in range(1, self.args.order + 1):
            adj_label = get_A_r(graph_org_torch, i)
            adj_label_list.append(adj_label)

        graph_org_torch = adj_label_list[0]
        adj_label = adj_label_list[-1]
        del adj_label_list

        test_acc_list = []
        loss_list =[]
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            embeds_tra, x_dis, C_loss = self.model(features, graph_org_torch)
            loss_Ncontrast = Ncontrast_v1(x_dis, adj_label, self.args.tau)
            loss_ce = F.cross_entropy(embeds_tra[self.idx_train], train_lbls)
            loss = loss_ce + loss_Ncontrast* self.args.alpha + C_loss * self.args.beta

            loss.backward(retain_graph=True)
            optimiser.step()
            loss_list.append(loss.item())

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.model.eval()
                embeds, _, _ = self.model(features, graph_org_torch)

                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)

                print("{:.4f}|".format(test_acc.item()),end="")

                test_acc_list.append(test_acc.item())

                index1 += 1
                if index1 % 10==0:
                    print("")
                    index1 = 0

                # early stop
                stop_epoch = epoch
                if val_acc >= best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break
            ################END|Eval|###############

        training_time = time.time() - start

        # print("")
        print("\n [Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))

        return output_acc, training_time, stop_epoch




def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)