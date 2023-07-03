import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib import ticker

from binbagnets.models import pytorchnet

model_names = sorted(name for name in pytorchnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(pytorchnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='binbagnet17',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         '(default: binbagnet17)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node '
                         'or multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help='use fake data to '
                                                         'benchmark')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    best_acc1 = 0
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = pytorchnet.__dict__[args.arch](pretrained=True)
        summary(model, input_size=(3, 224, 224), device="cpu")
    else:
        print("=> creating model '{}'".format(args.arch))
        model = pytorchnet.__dict__[args.arch]()
        summary(model, input_size=(3, 224, 224), device="cpu")

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current
                # node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) /
                                   ngpus_per_node)
                model = torch.nn.parallel\
                                .DistributedDataParallel(model,
                                                         device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size
                # to all available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.BCELoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # sets the learning rate to the initial LR decayed by 10 every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_path = os.path.normpath(os.path.join(args.resume, os.pardir,
                                         os.pardir, 'best_model.pth.tar'))
            best_checkpoint = torch.load(best_path,
                                         map_location=torch.device('cpu'))
            best_acc1 = best_checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000,
                                          transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000,
                                        transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed\
                             .DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed\
                           .DistributedSampler(val_dataset, shuffle=False,
                                               drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    train_metric_dict_base = dict()
    val_metric_dict_base = dict()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_metric_dict = train(train_loader, model, criterion, optimizer,
                                  epoch, device, args)

        # evaluate on validation set
        acc1, loss1, val_metric_dict = validate(val_loader, model, criterion,
                                                device, args)
        
        scheduler.step()
        
        train_metric_dict_base = append_metric_dicts(train_metric_dict_base,
                                                     train_metric_dict, epoch,
                                                     args.start_epoch)
        val_metric_dict_base = append_metric_dicts(val_metric_dict_base,
                                                   val_metric_dict, epoch,
                                                   args.start_epoch)
        save_metric_dict_curves(train_metric_dict_base, args.data, 'train',
                                args.start_epoch + 1, epoch + 1)
        save_metric_dict_curves(val_metric_dict_base, args.data, 'val',
                                args.start_epoch + 1, epoch + 1)
        save_base_metric_dict(train_metric_dict_base, args.data, 'train')
        save_base_metric_dict(val_metric_dict_base, args.data, 'val')
        
        # remember best acc@1, save checkpoint and save val acc curve
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        if not args.multiprocessing_distributed or\
           (args.multiprocessing_distributed and
                args.rank % ngpus_per_node == 0):
            save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(),
               'best_acc1': best_acc1
            }, is_best, args.data)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    acc_met = Accuracy(task="binary").to(device, non_blocking=True)
    prec_met = Precision(task="binary").to(device, non_blocking=True)
    rec_met = Recall(task="binary").to(device, non_blocking=True)
    f1_met = F1Score(task="binary").to(device, non_blocking=True)
    scores_met = StatScores(task="binary").to(device, non_blocking=True)
    metric_list_dict = dict()
    metric_list_dict['accuracy'] = []
    metric_list_dict['precision'] = []
    metric_list_dict['recall'] = []
    metric_list_dict['f1score'] = []
    metric_list_dict['scores'] = []
    metric_list_dict['loss'] = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device)

        # compute output
        output = model(images)
        output = torch.flatten(output)
        output = torch.sigmoid(output)
        loss = criterion(output, target.float())

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        
        # record train metrics to be saved
        metric_list_dict['accuracy'].append(acc_met(output, target).item())
        metric_list_dict['precision'].append(prec_met(output, target).item())
        metric_list_dict['recall'].append(rec_met(output, target).item())
        metric_list_dict['f1score'].append(f1_met(output, target).item())
        metric_list_dict['scores'].append(scores_met(output, target).cpu()
                                                                    .numpy())
        metric_list_dict['loss'].append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    
    # calculate average metrics for epochs
    metric_dict = dict()
    for key, value in metric_list_dict.items():
        if key == 'scores':
            metric_dict[key] = list(map(sum, zip(*value)))
        else:
            metric_dict[key] = mean(value)
    
    return metric_dict


def validate(val_loader, model, criterion, device, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                output = torch.flatten(output)
                output = torch.sigmoid(output)
                loss = criterion(output, target.float())

                # measure accuracy and record loss
                acc1 = accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                
                # record validation metrics to be saved
                metric_list_dict['accuracy'].append(acc_met(output, target)
                                                    .item())
                metric_list_dict['precision'].append(prec_met(output, target)
                                                     .item())
                metric_list_dict['recall'].append(rec_met(output, target)
                                                  .item())
                metric_list_dict['f1score'].append(f1_met(output, target)
                                                   .item())
                metric_list_dict['scores'].append(scores_met(output, target)
                                                  .cpu().numpy())
                metric_list_dict['loss'].append(loss.item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and 
                           (len(val_loader.sampler) * args.world_size <
                            len(val_loader.dataset))),
        [batch_time, losses, top1],
        prefix='Test: ')

    acc_met = Accuracy(task="binary").to(device, non_blocking=True)
    prec_met = Precision(task="binary").to(device, non_blocking=True)
    rec_met = Recall(task="binary").to(device, non_blocking=True)
    f1_met = F1Score(task="binary").to(device, non_blocking=True)
    scores_met = StatScores(task="binary").to(device, non_blocking=True)
    metric_list_dict = dict()
    metric_list_dict['accuracy'] = []
    metric_list_dict['precision'] = []
    metric_list_dict['recall'] = []
    metric_list_dict['f1score'] = []
    metric_list_dict['scores'] = []
    metric_list_dict['loss'] = []

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()

    if args.distributed and \
       (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * 
                                       args.world_size,
                                       len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    # calculate average metrics for epochs
    metric_dict = dict()
    for key, value in metric_list_dict.items():
        if key == 'scores':
            metric_dict[key] = list(map(sum, zip(*value)))
        else:
            metric_dict[key] = mean(value)

    progress.display_summary()

    return top1.avg, losses.avg, metric_dict


def save_checkpoint(state, is_best, dataset_path):
    models_folder = os.path.normpath(os.path.join(dataset_path, os.pardir,
                                                  'models'))
    all_models_folder = os.path.join(models_folder, 'all')
    if not os.path.exists(all_models_folder):
        os.makedirs(all_models_folder)
    model_path = os.path.join(all_models_folder,
                              'checkpoint_%d.pth.tar' % state['epoch'])
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(models_folder,
                                                 'best_model.pth.tar'))


def save_metric_curve(metric_list, metric_name, figs_folder, metric_params,
                      start_epoch, cur_epoch):
    x_lst = list(range(start_epoch, cur_epoch + 1))
    fig = plt.Figure()
    ax = fig.gca()
    ax.set_title('%s per epoch' % metric_params[0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_params[0])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.plot(x_lst, metric_list, metric_params[1])
    fig.savefig(os.path.join(figs_folder, '%s_curve.png' % metric_name))


def save_metric_dict_curves(metric_dict_base, dataset_path, subfolder,
                            start_epoch, cur_epoch):    
    metric_param_dict = dict()      # Y label      style
    metric_param_dict['accuracy']  = ['Accuracy',   'b-']
    metric_param_dict['precision'] = ['Precision',  'k-']
    metric_param_dict['recall']    = ['Recall',     'g-']
    metric_param_dict['f1score']   = ['F-1 Score',  'c-']
    metric_param_dict['loss']      = ['Loss value', 'r-']
    
    figs_folder = os.path.normpath(os.path.join(dataset_path, os.pardir,
                                                'figs', 'metrics', subfolder))
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)
    
    for key, value in metric_dict_base.items():
        if key != 'scores':
            save_metric_curve(value, key, figs_folder, metric_param_dict[key],
                              start_epoch, cur_epoch)


def save_base_metric_dict(metric_dict_base, dataset_path, subfolder):
    targ_folder = os.path.normpath(os.path.join(dataset_path, os.pardir,
                                                'figs', 'metrics', subfolder))
    if not os.path.exists(targ_folder):
        os.makedirs(targ_folder)
    targ_path = os.path.join(targ_folder, 'metric_dict.pth.tar')
    torch.save(metric_dict_base, targ_path)


def init_metric_dict_base(metric_dict_new):
    metric_dict_base = dict()
    for key in metric_dict_new:
        metric_dict_base[key] = []
    return metric_dict_base


def append_metric_dicts(metric_dict_base, metric_dict_new, cur_epoch,
                        start_epoch):
    if cur_epoch == start_epoch:
        metric_dict_base = init_metric_dict_base(metric_dict_new)
    for key in metric_dict_base:
        metric_dict_base[key].append(metric_dict_new[key])
    return metric_dict_base


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32,
                             device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target):
    """Computes the accuracy"""
    assert target.ndim == 1 and target.size() == output.size()
    y_prob = output > 0.5
    return (target == y_prob).sum().item() / target.size(0)


if __name__ == '__main__':
    main()
