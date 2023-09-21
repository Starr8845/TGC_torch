import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha, l4=False, weight_mask=[]):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    # return ratio's mse loss
    if len(weight_mask)==0:
        reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    else:
        reg_losses = (return_ratio * mask - ground_truth * mask) **2
        reg_loss = (reg_losses.mean() + (reg_losses*weight_mask*3).mean())/4
    # formula (4-6)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    if l4:
        # 为了long tail item 这里加一个l4 loss
        l4_loss = ((return_ratio * mask - ground_truth * mask)**4).mean()
        # 这里应该改成mean
        # print(loss, l4_loss)
        loss = loss + 10*l4_loss
    return loss, reg_loss, rank_loss, return_ratio




def cal_sample_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    # 拿到每个sample的loss
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    # return ratio's mse loss
    return_ratio_np = (return_ratio*mask).detach().cpu().numpy()
    sample_reg_loss = np.square((ground_truth*mask).cpu().numpy()-return_ratio_np).squeeze()
    # formula (4-6)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    sample_rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw),dim=0).detach().cpu().numpy()
    
    sample_loss = sample_reg_loss + alpha * sample_rank_loss
    return sample_loss, sample_reg_loss, sample_rank_loss

def cal_my_IC(prediction, label):
        pd_label = pd.Series(label.detach().cpu().numpy().squeeze())
        pd_prediction = pd.Series(prediction.detach().cpu().numpy().squeeze())
        return pd_prediction.corr(pd_label), pd_prediction.corr(pd_label, method="spearman")



class GraphModule(nn.Module):
    def __init__(self, batch_size, fea_shape, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = fea_shape
        self.inner_prod = inner_prod
        self.relation = nn.Parameter(torch.tensor(rel_encoding, dtype=torch.float32), requires_grad=False)
        self.rel_mask = nn.Parameter(torch.tensor(rel_mask, dtype=torch.float32), requires_grad=False)
        self.all_one = nn.Parameter(torch.ones(self.batch_size, 1, dtype=torch.float32), requires_grad=False)
        self.rel_weight = nn.Linear(rel_encoding.shape[-1], 1)
        if self.inner_prod is False:
            self.head_weight = nn.Linear(fea_shape, 1)
            self.tail_weight = nn.Linear(fea_shape, 1)

    def forward(self, inputs):
        rel_weight = self.rel_weight(self.relation)
        if self.inner_prod:
            inner_weight = inputs @ inputs.t()
            weight = inner_weight @ rel_weight[:, :, -1]
        else:
            all_one = self.all_one
            head_weight = self.head_weight(inputs)
            tail_weight = self.tail_weight(inputs)
            weight = (head_weight @ all_one.t() + all_one @ tail_weight.t()) + rel_weight[:, :, -1]
        weight_masked = F.softmax(self.rel_mask + weight, dim=0)
        outputs = weight_masked @ inputs
        return outputs


class StockLSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        x, _ = self.lstm_cell(inputs)
        x = x[:, -1, :]
        prediction = F.leaky_relu(self.fc(x))
        return prediction


class RelationLSTM(nn.Module):
    def __init__(self, batch_size, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.graph_layer = GraphModule(batch_size, 64, rel_encoding, rel_mask, inner_prod)
        self.fc = nn.Linear(64 * 2 + 5, 1)
        self.fc_residual = nn.Linear(5, 5)

    def forward(self, inputs):
        # inputs: [1026, 16, 5]
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]   # [1026,64]
        outputs_graph = self.graph_layer(x)  # [1026,64]
        # outputs_cat = torch.cat([x, outputs_graph], dim=1)
        residual = self.fc_residual(inputs[:,-1,:].squeeze())
        outputs_cat = torch.cat([residual, x, outputs_graph], dim=1)
        prediction = F.leaky_relu(self.fc(outputs_cat))
        return prediction
