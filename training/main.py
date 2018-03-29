import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')

from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training
from torch import nn
import math
from layers import acc
import multiprocessing as mp

from layers import acc

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')


def mp_get_pr(conf_th, nms_th, detect_th, pbb, lbb, num_procs=4):
    start_time = time.time()

    num_samples = len(pbb)
    split_size = int(np.ceil(float(num_samples) / num_procs))
    num_procs = int(np.ceil(float(num_samples) / split_size))

    manager = mp.Manager()
    tp = manager.list(range(num_procs))
    fp = manager.list(range(num_procs))
    fn = manager.list(range(num_procs))
    procs = []
    for pid in range(num_procs):
        proc = mp.Process(
            target=get_pr,
            args=(
                pbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                lbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                conf_th, nms_th, detect_th, pid, tp, fp, fn))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    tp_rate = float(tp) / (tp + fn)
    fp_rate = float(tp) / (tp + fp)

    end_time = time.time()
    print('conf_th %1.1f, nms_th %1.1f, detect_th %1.1f, tp %d, fp %d, fn %d, tp rate %f, fp rate %f, time %3.2f' %
          (conf_th, nms_th, detect_th, tp, fp, fn, tp_rate, fp_rate, end_time - start_time))

    return tp_rate, fp_rate


def get_pr(pbb, lbb, conf_th, nms_th, detect_th, pid, tp_list, fp_list, fn_list):
    tp, fp, fn = 0, 0, 0
    for i in range(len(pbb)):
        tpi, fpi, pi, _ = acc(pbb[i], lbb[i], conf_th, nms_th, detect_th)
        tp = tp + len(tpi)
        fp = fp + len(fpi)
        fn = fn + len(pi)
    tp_list[pid] = tp
    fp_list[pid] = fp
    fn_list[pid] = fn


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    print ("arg", args.gpu)
    print ("num_gpu", n_gpu)

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path']
    print ("datadir", datadir)
    print ("anchor", config['anchors'])
    print ("pad_val", config['pad_value'])
    print ("th_pos_train", config['th_pos_train'])

    if args.test == 1:
        margin = 32
        sidelen = 144
        print ("args.test True")
        split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
        dataset = data.DataBowl3Detector(
            datadir,
            'val9.npy',
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=data.collate,
            pin_memory=False)

        test(test_loader, net, get_pbb, save_dir, config, sidelen)
        return

    # net = DataParallel(net)

    train_dataset = data.DataBowl3Detector(
        datadir,
        'train_luna_9.npy',
        config,
        phase='train')
    print ("len train_dataset", train_dataset.__len__())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_dataset = data.DataBowl3Detector(
        datadir,
        'val9.npy',
        config,
        phase='val')
    print ("len val_dataset", val_dataset.__len__())

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    margin = 32
    sidelen = 144

    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
    test_dataset = data.DataBowl3Detector(
        datadir,
        'val9.npy',
        config,
        phase='test',
        split_comber=split_comber)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data.collate,
        pin_memory=False)

    print ("lr", args.lr)
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    best_val_loss = 100
    best_test_loss = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print ("epoch", epoch)
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
        best_val_loss = validate(val_loader, net, loss, best_val_loss, epoch, save_dir)
        if ((epoch > 150) and ((epoch + 1) % 10) == 0):
            best_test_loss = test_training(test_loader, net, get_pbb, save_dir, config, sidelen, best_test_loss, epoch, n_gpu)

        if ((epoch > 300) and ((epoch + 1) % 100) == 0):
            num_neg = train_dataset.get_neg_num_neg() + 800
            train_dataset.set_neg_num_neg(num_neg)



def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        coord = Variable(coord.cuda(async=True))
        output = net(data, coord)
        # print ("output", np.shape(output))
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print


def validate(data_loader, net, loss, best_val_loss, epoch, save_dir):
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)
        coord = Variable(coord.cuda(async=True), volatile=True)

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    val_loss = np.mean(metrics[:, 1])

    if (val_loss < best_val_loss):
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, 'best.ckpt'))

        best_val_loss = val_loss

    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print ('best_val_loss ', best_val_loss)

    return best_val_loss


def test_training(data_loader, net, get_pbb, save_dir, config, sidelen, best_test_loss, epoch, n_gpu):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net.eval()
    namelist = []

    pbb_list = []
    lbb_list = []

    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]

        n_per_run = n_gpu

        splitlist = list(range(0, len(data) + 1, n_per_run))

        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
            output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)

        e = time.time()
        pbb_list.append(pbb)
        lbb_list.append(lbb)

    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))

    ct = 1
    nt = 0.3
    dt = 0.3
    pbb_list = np.array(pbb_list)
    lbb_list = np.array(lbb_list)
    tp_rate, fp_rate = mp_get_pr(1, 0.3, 0.3, pbb_list, lbb_list)
    test_loss = tp_rate + fp_rate

    if (test_loss > best_test_loss):
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, 'best_validate_all.ckpt'))

        best_test_loss = test_loss
    print ("best_test_loss", best_test_loss)

    return best_test_loss


def test(data_loader, net, get_pbb, save_dir, config, sidelen):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        print ("i_name", i_name)
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]

        n_per_run = args.n_test

        splitlist = list(range(0, len(data) + 1, n_per_run))

        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
            output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)

        print([i_name, name])
        e = time.time()

        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)

    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data, config['max_stride'], margin)
    data = Variable(data.cuda(async=True), volatile=True, requires_grad=False)
    splitlist = range(0, args.split + 1, n_per_run)
    outputlist = []

    for i in range(len(splitlist) - 1):
        output = net(data[splitlist[i]:splitlist[i + 1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)

    output = np.concatenate(outputlist, 0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    return output


if __name__ == '__main__':
    main()

