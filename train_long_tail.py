import random
import numpy as np
import os
import torch as torch
from load_data import load_relation_data, load_EOD_data
from evaluator import evaluate
from model import get_loss, RelationLSTM, cal_my_IC, StockLSTM, cal_sample_loss, contrastive_three_modes_loss
from utils import string_format, Logger
from tsne import plot_embedding, plot_embedding_heatmap, cal_distance
import matplotlib.pyplot as plt


def validate(start_index, end_index, long_tail_masks=[]):
    """
    get loss on validate/test set
    """
    my_ICs = []
    my_RICs = []
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - parameters['seq'] - steps + 1, end_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     batch_size, parameters['alpha'])
            IC, RIC = cal_my_IC(cur_rr * mask_batch, gt_batch * mask_batch)
            my_ICs.append(IC)
            my_RICs.append(RIC)
            
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        IC, ICIR, RIC, RICIR =  np.nanmean(my_ICs), np.nanmean(my_ICs)/np.nanstd(my_ICs), np.nanmean(my_RICs), np.nanmean(my_RICs)/np.nanstd(my_RICs)
        cur_valid_perf.update({"IC":IC,"ICIR":ICIR,"RIC":RIC,"RICIR":RICIR})
        if len(long_tail_masks)>0:
            long_tail_result = []
            for long_tail_mask in long_tail_masks:
                long_tail_result.append(np.linalg.norm((cur_valid_pred - cur_valid_gt) * long_tail_mask) ** 2 / np.sum(long_tail_mask))
            cur_valid_perf["long_tail"] = long_tail_result
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_difficulty_mask(base_model, start_index, end_index):
    # 把样本划分成1%难, 5%难, 10%难, 20%难的组
    # 需要一个基准模型   这个基准模型用LSTM或RSR
    # 要先做一遍inference  得到所有的样本的loss
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), end_index - start_index], dtype=float)
        for cur_offset in range(start_index - parameters['seq'] - steps + 1, end_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = base_model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     batch_size, parameters['alpha'])
            cur_valid_pred[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = mask_batch[:, 0].cpu()
    
    mask1, mask5, mask10, mask20 = split_by_difficulty(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return mask1, mask5, mask10, mask20, ((cur_valid_pred - cur_valid_gt) * cur_valid_mask) ** 2 


def split_by_difficulty(cur_valid_pred, cur_valid_gt, cur_valid_mask):
    h, w = cur_valid_pred.shape
    loss = np.square(cur_valid_gt*cur_valid_mask-cur_valid_pred*cur_valid_mask)
    sort_index = np.argsort(loss.reshape(-1))[::-1]
    sort_index_1 = sort_index//w
    sort_index_2 = sort_index%w
    mask1 = np.zeros([h, w], dtype=float)
    mask1[sort_index_1[:int(0.01*h*w)], sort_index_2[:int(0.01*h*w)]] = 1
    mask1 = mask1*cur_valid_mask
    mask5 = np.zeros([h, w], dtype=float)
    mask5[sort_index_1[:int(0.05*h*w)], sort_index_2[:int(0.05*h*w)]] = 1
    mask5 = mask5*cur_valid_mask
    mask10 = np.zeros([h, w], dtype=float)
    mask10[sort_index_1[:int(0.1*h*w)], sort_index_2[:int(0.1*h*w)]] = 1
    mask10 = mask10*cur_valid_mask
    mask20 = np.zeros([h, w], dtype=float)
    mask20[sort_index_1[:int(0.2*h*w)], sort_index_2[:int(0.2*h*w)]] = 1
    mask20 = mask20*cur_valid_mask
    return mask1, mask5, mask10, mask20



def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = parameters['seq']
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

def train(long_tail_masks={"train":[],"valid":[],"test":[]}, long_tail_scores={"train":[]}):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # lr 从1e-3改到1e-4
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    batch_offsets = np.arange(start=0, stop=valid_index- parameters['seq'] - steps + 1, dtype=int)
    print(long_tail_scores['train'].shape)
    print(batch_offsets.shape)
    # train loop
    for epoch in range(epochs):
        logging.info(f"EPOCH {epoch}")
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        # steps
        for j in range(valid_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            pred_repre = model.get_repre(data_batch) # [1026, 133]
            prediction = model.predict(pred_repre) # 
            # l4 loss or reweighting
            weight_mask = torch.Tensor(long_tail_masks["train"][-3][:, batch_offsets[j]].squeeze()).to(device) # 放大20%的样本的loss
            # cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                # batch_size, parameters['alpha'], l4=False, weight_mask=weight_mask) 
            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                batch_size, parameters['alpha'], l4=False, weight_mask=[]) 
            # 先尝试在一个batch内部进行拉近和拉远
            scores = torch.Tensor(long_tail_scores['train'][:, batch_offsets[j]].squeeze()).to(device)
            # if epoch>10:
            #     contrastive_loss, pos_num, neg_num = contrastive_three_modes_loss(features=pred_repre, scores=scores, weight_mask=weight_mask, temp=0.1, base_temperature=0.07)
            #     # print(pos_num, neg_num)
            #     # update model
            #     cur_loss += contrastive_loss/1000  #根据测试 /1000效果不如/100
            cur_loss.backward()
            optimizer.step()

            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()

        # train loss
        # loss = reg_loss(mse) + alpha*rank_loss
        tra_loss = tra_loss / (valid_index - parameters['seq'] - steps + 1)
        tra_reg_loss = tra_reg_loss / (valid_index - parameters['seq'] - steps + 1)
        tra_rank_loss = tra_rank_loss / (valid_index - parameters['seq'] - steps + 1)
        logging.info('Train : loss:{} reg_loss:{} rank_loss:{}'.format(string_format(tra_loss), string_format(tra_reg_loss), string_format(tra_rank_loss)))

        tra_loss, tra_reg_loss, tra_rank_loss, tra_perf = validate(parameters['seq'], valid_index, long_tail_masks["train"])
        logging.info('train : loss:{} reg_loss:{} rank_loss:{}'.format(string_format(tra_loss), string_format(tra_reg_loss), string_format(tra_rank_loss)))
        logging.info(f'Train performance:{string_format(tra_perf)}', )
        # show performance on valid set
        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index, long_tail_masks["valid"])
        logging.info('Valid : loss:{} reg_loss:{} rank_loss:{}'.format(string_format(val_loss), string_format(val_reg_loss), string_format(val_rank_loss)))
        logging.info(f'Valid performance:{string_format(val_perf)}', )

        # show performance on valid set
        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates, long_tail_masks["test"])
        logging.info('Test: loss:{} reg_loss:{} rank_loss:{}'.format(string_format(test_loss), string_format(test_reg_loss), string_format(test_rank_loss)))
        logging.info(f'Test performance:{string_format(test_perf)}')

        # best result
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
            logging.info(f'Better valid loss:{string_format(best_valid_loss)}')
            torch.save(model.state_dict(), f"{log_folder_path}/{epoch}.pt")
    logging.info(f'\nBest Valid performance:{string_format(best_valid_perf)}')
    logging.info(f'Best Test performance:{string_format(best_test_perf)}')


def draw_tsne(long_tail_masks={"train":[],"valid":[],"test":[]}, long_tail_scores={"train":[]}):
    batch_offsets = np.arange(start=0, stop=valid_index- parameters['seq'] - steps + 1, dtype=int)
    with torch.no_grad():
        for j in range(valid_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            pred_repre = model.get_repre(data_batch) # [1026, 133]
            weight_mask = long_tail_masks["train"][-3][:, batch_offsets[j]].squeeze() # 放大5%的样本的loss
            fig = plot_embedding(data=pred_repre.cpu().numpy(), label=weight_mask, title='try')
            plt.savefig(f"{log_folder_path}/top5_{j}.png")
            plt.close()
            fig = plot_embedding_heatmap(data=pred_repre.cpu().numpy(), label=gt_batch.cpu().numpy().squeeze(),mask=weight_mask, title='pred_label')
            plt.savefig(f'{log_folder_path}/pred_label_{j}.png')
            plt.close()
            hard_avg, easy_avg, all_avg = cal_distance(pred_repre.cpu().numpy(), weight_mask)
            logging.info('day:{} hard_avg:{} easy_avg:{} all_avg:{}'.format(j, string_format(hard_avg), string_format(easy_avg), string_format(all_avg)))
            if j>20:
                exit(-1)
    

if __name__=="__main__":
    np.random.seed(123456789)
    torch.random.manual_seed(12345678)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    myLogger = Logger(info="")
    log_folder_path, _ = myLogger.get_log_path()
    from utils import logging

    data_path = 'data/2013-01-01'
    market_name = 'NASDAQ'
    relation_name = 'wikidata'  # or sector_industry
    parameters = {'seq': 16, 'unit': 64, 'alpha': 0.1}
    epochs = 50
    valid_index = 756
    test_index = 1008
    fea_dim = 5
    steps = 1

    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname), dtype=str, delimiter='\t', skip_header=False)
    batch_size = len(tickers)
    logging.info(f'#tickers selected:{len(tickers)}', )
    eod_data, mask_data, gt_data, price_data = load_EOD_data(data_path, market_name, tickers, steps)
    trade_dates = mask_data.shape[1]
    # relation data
    rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
    rel_encoding, rel_mask = load_relation_data(
        os.path.join(data_path, '..', 'relation', relation_name, market_name + rname_tail[relation_name])
    )
    logging.info(f'relation encoding shape:{rel_encoding.shape}')
    logging.info(f'relation mask shape:{rel_mask.shape}')

    model = RelationLSTM(
        batch_size=batch_size,
        rel_encoding=rel_encoding,
        rel_mask=rel_mask
    ).to(device)
    # model = StockLSTM(
    #     batch_size=batch_size
    # ).to(device)
    
    base_model = StockLSTM(
        batch_size=batch_size
    ).to(device) 
    base_model.load_state_dict(torch.load("/home/zzx/quant/TOIS19_pytorch/TGC_torch/logs/BaseModels/StockLSTM/23.pt"))
    # 测试集上
    train_mask1, train_mask5, train_mask10, train_mask20, train_scores = get_difficulty_mask(base_model, parameters['seq'], valid_index)
    valid_mask1, valid_mask5, valid_mask10, valid_mask20, valid_scores = get_difficulty_mask(base_model, valid_index, test_index)
    test_mask1, test_mask5, test_mask10, test_mask20, test_scores = get_difficulty_mask(base_model, test_index, trade_dates)
    long_tail_masks = {"train":[train_mask1, train_mask5, train_mask10, train_mask20],
                           "valid":[valid_mask1, valid_mask5, valid_mask10, valid_mask20],
                           "test":[test_mask1, test_mask5, test_mask10, test_mask20],}
    
    # train(long_tail_masks=long_tail_masks,long_tail_scores={"train":train_scores})
    model.load_state_dict(torch.load(f"/home/zzx/quant/TOIS19_pytorch/TGC_torch/logs/BaseModels/RelationLSTM/44.pt"))
    draw_tsne(long_tail_masks=long_tail_masks, long_tail_scores={"train":train_scores})

