import random
import numpy as np
import os
import torch as torch
from load_data import load_relation_data, load_EOD_data
from evaluator import evaluate
from model import get_loss, RelationLSTM, cal_sample_loss
from analysis import draw_one_day_loss


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

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
print('#tickers selected:', len(tickers))
eod_data, mask_data, gt_data, price_data = load_EOD_data(data_path, market_name, tickers, steps)
trade_dates = mask_data.shape[1]
# relation data
rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
rel_encoding, rel_mask = load_relation_data(
    os.path.join(data_path, '..', 'relation', relation_name, market_name + rname_tail[relation_name])
)
print('relation encoding shape:', rel_encoding.shape)
print('relation mask shape:', rel_mask.shape)

model = RelationLSTM(
    batch_size=batch_size,
    rel_encoding=rel_encoding,
    rel_mask=rel_mask
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


def validate(start_index, end_index):
    """
    get loss on validate/test set
    """
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
    return loss, reg_loss, rank_loss, cur_valid_perf


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


# train loop
for epoch in range(epochs):
    print(f"\n\nepoch:{epoch}")
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
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            batch_size, parameters['alpha'])
        # update model
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()

        # # 每个batch的数据
        # sample_loss, sample_reg_loss, sample_rank_loss = cal_sample_loss(prediction, gt_batch, price_batch, mask_batch,
        #                                                     batch_size, parameters['alpha'])
        # draw_one_day_loss(sample_loss,f"/home/zzx/quant/TOIS19_pytorch/TGC_torch/figs/{j}-sample_loss.jpg")
        # draw_one_day_loss(sample_loss,f"/home/zzx/quant/TOIS19_pytorch/TGC_torch/figs/{j}-reg_loss.jpg")
        # draw_one_day_loss(sample_loss,f"/home/zzx/quant/TOIS19_pytorch/TGC_torch/figs/{j}-rank_loss.jpg")
        # if j>5:
        #     break
    # train loss
    # loss = reg_loss(mse) + alpha*rank_loss
    tra_loss = tra_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - parameters['seq'] - steps + 1)
    print('Train : loss:{%.6f} reg_loss:{%.6f} rank_loss:{%.6f}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    # show performance on valid set
    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{%.6f} reg_loss:{%.6f} rank_loss:{%.6f}'.format(val_loss, val_reg_loss, val_rank_loss))
    print('\t Valid performance:', val_perf)

    # show performance on valid set
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{%.6f} reg_loss:{%.6f} rank_loss:{%.6f}'.format(test_loss, test_reg_loss, test_rank_loss))
    print('\t Test performance:%.6f'%( test_perf))

    # best result
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        # In this place, remove some var that wouldn't be printed
        # without copy.copy()
        best_valid_perf = val_perf
        best_test_perf = test_perf
        print('Better valid loss:%.6f'%( best_valid_loss))
        torch.save(model.state_dict(), f"/home/zzx/quant/TOIS19_pytorch/TGC_torch/figs/{epoch}.pt")
print('\nBest Valid performance:%.6f'%( best_valid_perf))
print('Best Test performance:%.6f'%(best_test_perf))
