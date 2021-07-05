import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
from training.load_data import load_EOD_data, get_batch, get_fund_adj_H, get_industry_adj

import pandas as pd

fund_adj_tensor_H = get_fund_adj_H()
adj = torch.Tensor(get_industry_adj())
eod_data, ground_truth = load_EOD_data()

fund_adj_tensor_H = torch.Tensor(fund_adj_tensor_H)

Htensor2 = torch.randn(0)
for i in range(28):
    fund = fund_adj_tensor_H[i]

for i in range(28):
    Htensor = torch.randn(0)
    for j in range(61):
        Htensor = torch.cat([Htensor, torch.Tensor(fund_adj_tensor_H[i]).unsqueeze(0)], dim=0)

    Htensor2 = torch.cat([Htensor2, torch.Tensor(Htensor)], dim=0)

def cal_performance(pred, gold, smoothing=False):

    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    percision = metrics.precision_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    recall = metrics.recall_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    f1_score = metrics.f1_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='weighted')

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct, percision, recall, f1_score

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = 3

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss

def train_epoch(model,training_data, optimizer, device, smoothing,args):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    total_accu = 0
    n_count = 0
    for step, (eod, gt) in enumerate(training_data):

        H = Htensor2[args.batch_size*step]
        Eod, Gt, H_,adj_= eod.to(device), gt.to(device), H.to(device), adj.to(device)

        # forward
        optimizer.zero_grad()
        pred = model(Eod,H_,adj_,args.hidden)

        # backward
        loss, n_correct, percision, recall, f1_score = cal_performance(pred, Gt, smoothing=smoothing)
        loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_accu += n_correct
        n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy, percision, recall, f1_score


def eval_epoch(model, validation_data, device,args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_accu = 0
    n_count = 0
    valid_pred = []
    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            H = Htensor2[args.train_index+args.batch_size * step]

            # prepare data
            Eod, Gt, H_, adj_,= eod.to(device), gt.to(device), H.to(device), adj.to(device)

            # forward
            pred = model(Eod, H_, adj_, args.hidden)
            loss, n_correct, percision, recall, f1_score = cal_performance(pred, Gt, smoothing=False)
            pred = pred.max(1)[1]
            pred = pred.cuda().data.cpu().numpy()
            valid_pred.extend(pred)

            total_loss += loss.item()
            total_accu += n_correct
            n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy, percision, recall, f1_score, valid_pred


def train(model, training_data, validation_data, optimizer, device, args):
    ''' Start training '''


    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss,  accuracy,percision,recall,f1_score\n')
            log_vf.write('epoch, loss,  accuracy,percision,recall,f1_score\n')

    valid_accus = []
    Train_Loss_list = []
    Train_Accuracy_list = []
    Val_Loss_list = []
    Val_Accuracy_list = []
    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_percision, train_recall, train_f1_score = train_epoch(
            model, training_data, optimizer, device, smoothing=args.label_smoothing, args=args)

        Train_Loss_list.append(train_loss)
        Train_Accuracy_list.append(100 * train_accu)

        print(
            ' - (Training) loss:{loss:8.5f}, accuracy:{accu:3.3f}%,  percision:{perc:3.3f}%,  recall:{recall:3.3f}%,  f1_score:{f1:3.3f}% , ' \
            'elapse: {elapse:3.3f} min'.format(
                loss=train_loss, accu=100 * train_accu,
                perc=100 * train_percision, recall=100 * train_recall, f1=100 * train_f1_score,
                elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu, valid_percision, valid_recall, valid_f1_score, valid_pred = eval_epoch(model, validation_data,
                                                                                           device, args=args)
        Val_Loss_list.append(valid_loss)
        Val_Accuracy_list.append(100 * valid_accu)
        print(
            ' - (Validation) loss:{loss:8.5f}, accuracy:{accu:3.3f}% ,  percision:{perc:3.3f}%,  recall:{recall:3.3f}%,  f1_score:{f1:3.3f}%, ' \
            'elapse: {elapse:3.3f} min'.format(
                loss=valid_loss, accu=100 * valid_accu,
                perc=100 * valid_percision, recall=100 * valid_recall, f1=100 * valid_f1_score,
                elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
                    dataframe = pd.DataFrame(valid_pred)
                    dataframe.to_csv('../10_days/valid_predict.csv', index=False, header=False, encoding='utf-8')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(
                    '{epoch: 4.0f},{loss: 8.5f},{accu:3.3f},  {perc:3.3f},  {recall:3.3f},  {f1:3.3f} \n'.format(
                        epoch=epoch_i, loss=train_loss, accu=100 * train_accu, perc=100 * train_percision,
                        recall=100 * train_recall, f1=100 * train_f1_score))
                log_vf.write(
                    '{epoch: 4.0f},{loss: 8.5f},{accu:3.3f},  {perc:3.3f},  {recall:3.3f},  {f1:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss, accu=100 * valid_accu, perc=100 * valid_percision,
                        recall=100 * valid_recall, f1=100 * valid_f1_score))

    if log_valid_file:
        with open(log_valid_file, 'a') as log_vf:
            log_vf.write('{Best:3.3f}\n'.format(Best=100 * max(valid_accus)))
            log_vf.write('{Best_epoch: 4.0f}\n'.format(Best_epoch=valid_accus.index(max(valid_accus))))

    # plt
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18.5, 10.5))
    x1 = x2 = x3 = x4 = range(args.epoch)
    y1 = Train_Loss_list
    y2 = Train_Accuracy_list
    y3 = Val_Loss_list
    y4 = Val_Accuracy_list
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(2, 2, 2)
    plt.plot(x2, y2, '.-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 2, 3)
    plt.plot(x3, y3, 'o-')
    plt.title('Val loss vs. epoches')
    plt.ylabel('Val loss')
    plt.subplot(2, 2, 4)
    plt.plot(x4, y4, 'o-')
    plt.title('Val accuracy vs. epoches')
    plt.ylabel('Val accuracy')
    # plt.show()


def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []

    for i in range(eod_data.shape[1] - args.length):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:args.valid_index], GT[args.train_index:args.valid_index]
    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod)
    train_gt, valid_gt, test_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt), torch.LongTensor(test_gt)


    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=True)


    return train_loader, valid_loader