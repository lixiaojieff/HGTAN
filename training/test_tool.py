import torch
import torch.nn.functional as F
import time
from sklearn import metrics
from training.load_data import load_EOD_data, get_batch, get_fund_adj_H, get_industry_adj
import torch.utils.data as Data
import os


fund_adj_tensor_H = get_fund_adj_H()
adj = torch.Tensor(get_industry_adj())
eod_data, ground_truth = load_EOD_data()

fund_adj_tensor_H = torch.Tensor(fund_adj_tensor_H)

Htensor2 = torch.randn(0)
for i in range(28):
    Htensor = torch.randn(0)
    for j in range(61):
        Htensor = torch.cat([Htensor, torch.Tensor(fund_adj_tensor_H[i]).unsqueeze(0)], dim=0)
    Htensor2 = torch.cat([Htensor2, torch.Tensor(Htensor)], dim=0)


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

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
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')

    return loss

def eval_epoch(model, test_data, device,args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_accu = 0
    n_count = 0
    test_pred = []
    test_pred_before = []
    with torch.no_grad():
        for step, (eod, gt) in enumerate(test_data):

            H = Htensor2[args.valid_index + args.batch_size * step]

            Eod, Gt, H_, adj_, = eod.to(device), gt.to(device), H.to(device), adj.to(device)
            pred = model(Eod, H_, adj_, args.hidden)

            loss, n_correct, percision, recall, f1_score = cal_performance(pred, Gt, smoothing=False)
            pred = F.softmax(pred, dim=1)
            pred = pred.cuda().data.cpu().numpy()
            test_pred_before.append(pred)
            pred = torch.Tensor(pred).to(device)
            pred = pred.max(1)[1]

            test_pred.extend(pred)

            total_loss += loss.item()
            total_accu += n_correct
            n_count += Eod.size(0) * Eod.size(1)

    epoch_loss = total_loss / n_count
    accuracy = total_accu / n_count
    return epoch_loss, accuracy, percision, recall, f1_score, test_pred


def train(model, test_data, optimizer, device, args):

    log_dir = args.save_model + '.chkpt'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        # args.load_state_dict(checkpoint['settings'])
        start_epoch = checkpoint['epoch']
        print('load epoch {} successfully！'.format(start_epoch))
    else:

        print('no save model, start training from the beginning！')
    start = time.time()
    test_loss, test_accu, test_percision, test_recall, test_f1_score, test_pred = eval_epoch(model, test_data,
                                                                                           device,args)
    print(
        ' - (test) loss:{loss:8.5f}, accuracy:{accu:3.3f}% ,  percision:{perc:3.3f}%,  recall:{recall:3.3f}%,  f1_score:{f1:3.3f}%, ' \
        'elapse: {elapse:3.3f} min'.format(
            loss=test_loss, accu=100 * test_accu,
            perc=100 * test_percision, recall=100 * test_recall, f1=100 * test_f1_score,
            elapse=(time.time() - start) / 60))



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

    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(
        test_eod)
    train_gt, valid_gt, test_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt), torch.LongTensor(test_gt)


    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)
    test_dataset = Data.TensorDataset(test_eod, test_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True)


    return train_loader, valid_loader, test_loader