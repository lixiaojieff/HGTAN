import argparse
import numpy as np
from training.load_data import load_EOD_data
import torch
from HGTAN.models import Transformer
from HGTAN.Optim import ScheduledOptim
# from training.tool import prepare_dataloaders, train
from training.test_tool import prepare_dataloaders,train
import torch.optim as optim

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
start = time.time()
def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', default=20,
                        help='length of historical sequence for feature')
    parser.add_argument('-train_index', type=int, default=1021)  # 0.6
    parser.add_argument('-valid_index', type=int, default=1361)  # 0.2
    parser.add_argument('-epoch', type=int, default=600)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('--rnn_unit', type=int, default=32, help='Number of hidden units.')  #######
    parser.add_argument('-d_model', type=int, default=16)  # d_k=d_v=d_model/n_head

    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')  #######
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-proj_share_weight', default='True')
    # parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default='../20_days_6/lstm+trans+HGAT3_5_valid1')
    parser.add_argument('-save_model', default='../20_days_6/lstm+HGAT3_5_valid1')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')
    parser.add_argument('-path', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-market', help='market name', default='NASDAQ')
    parser.add_argument('-tickers', help='fname for selected tickers')
    parser.add_argument('-threshold', type=float, default=0.005,
                        help='NASDAQ threshold')
    # parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()


    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    eod_data, ground_truth = load_EOD_data()
    train_loader, valid_loader, test_loader = prepare_dataloaders(eod_data, ground_truth, args)

    # ========= Preparing Model =========#
    print(args)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = Transformer(
        args.length,
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,

        n_head=args.n_head,
        dropout=args.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(model, test_loader, optimizer, device, args)
end = time.time()
print('time:',end-start)
if __name__ == '__main__':
    main()