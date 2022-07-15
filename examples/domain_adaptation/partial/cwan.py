import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from common.modules.classifier import Classifier
from dalib.partial_adaptation.cwan import CycleInconsistencyWeightModuleEuclidean, DomainAdversarialLoss, ImageClassifier, center_loss
import common.vision.datasets.partial as datasets
from common.vision.datasets.partial import default_partial as partial
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

import numpy as np
from sklearn.cluster import KMeans
from dalib.modules.entropy import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    partial_dataset = partial(dataset)
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = partial_dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=val_transform)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_target_dataset = partial_dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    val_dataset = partial_dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = partial_dataset(root=args.root, task=args.target, split='test', download=True,
                                       transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    num_classes = train_source_dataset.num_classes
    backbone = models.__dict__[args.arch](pretrained=True)

    if args.data == 'ImageNetCaltech':
        classifier = Classifier(backbone, num_classes, head=backbone.copy_head()).to(device)
    else:
        classifier = ImageClassifier(backbone, num_classes, args.bottleneck_dim, bias=not args.no_classifier_bias).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # 
    # pre-processing
    # 
    if args.pretrain != '':
        pretrain_pth = osp.join(args.pretrain, args.log.split('/')[-1], 'checkpoints', 'best.pth')
        print("=> using pre-trained model checkpoint path '{}'".format(pretrain_pth))
        classifier.load_state_dict(torch.load(pretrain_pth))

    # clustering src domain
    acc1, src_feature_np, src_target_np = get_feature(val_source_loader, classifier, args)
    print("val_src_acc1 = {:3.6f} and clustering feature ndarray shape is {}".format(acc1, src_feature_np.shape))
    src_index_np = np.zeros((src_feature_np.shape[0], num_classes), dtype=np.float32)
    src_index_np[np.array(list(range(src_feature_np.shape[0]))), src_target_np] = 1.0
    src_index_np /= src_index_np.sum(axis=0, keepdims=True)
    src_center = np.matmul(src_index_np.transpose((1,0)), src_feature_np)
    src_center = torch.from_numpy(src_center).to(device)
    print('source center shape: {}'.format(src_center.shape))

    # clustering trg domain
    acc1, trg_feature_np, _ = get_feature(val_target_loader, classifier, args)
    print("val_trg_acc1 = {:3.6f} and clustering feature ndarray shape is {}".format(acc1, trg_feature_np.shape))
    trg_kmeans = KMeans(n_clusters=args.num_cluster, random_state=args.seed).fit(trg_feature_np)
    trg_center = torch.from_numpy(trg_kmeans.cluster_centers_).to(device)
    print('target center shape: {}'.format(trg_center.shape))

    cwan_module = CycleInconsistencyWeightModuleEuclidean(softmax_temp=args.similarity_temp, num_cluster=args.num_cluster, 
                  src_cluster=src_center, trg_cluster=trg_center, momentum=args.cluster_momentum)


    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, cwan_module, domain_adv, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.6f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.6f}".format(acc1))

    logger.close()


def get_entropy_weight(score):
    weight = 1.0 + torch.exp(-entropy(score))
    weight = weight / weight.mean()
    return weight.detach()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, cwan_module: CycleInconsistencyWeightModuleEuclidean, 
          domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.6f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.6f')
    domain_accs = AverageMeter('Domain Acc', ':3.6f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, optimizer.param_groups[-1]['lr'], losses, cls_accs, tgt_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        # transform
        # src->trg->src
        # non-cycle
        f_s_t = cwan_module.xdomain_transform(f_s, cwan_module.trg_cluster.clone().detach())
        y_s_t = model.forward_feature(f_s_t)
        w_s_n = cwan_module.get_importance_weight(F.softmax(y_s_t, dim=1), labels_s)
        # cycle
        f_s_h = cwan_module.xdomain_transform(f_s_t, cwan_module.src_cluster.clone().detach())
        y_s_h = model.forward_feature(f_s_h)
        w_s_c = cwan_module.get_importance_weight(F.softmax(y_s_h, dim=1), labels_s)
        # combination
        e_w_n = get_entropy_weight(F.softmax(y_s_t, dim=1))
        w_s = w_s_n * args.mixing_factor * e_w_n + w_s_c * (1 - args.mixing_factor) 
        w_s = w_s / w_s.mean()
        w_s = w_s.detach()

        # trg->src->trg
        f_t_s = cwan_module.xdomain_transform(f_t, cwan_module.src_cluster.clone().detach())
        f_t_h = cwan_module.xdomain_transform(f_t_s, cwan_module.trg_cluster.clone().detach())
        y_t_h = model.forward_feature(f_t_h)

        # loss 
        # ce
        cls_loss = F.cross_entropy(y_s, labels_s, reduction='none')
        cls_loss = torch.mean(cls_loss * w_s, dim=0, keepdim=True) 
        # center loss
        src_ctr_loss = center_loss(f_s, cwan_module.src_cluster.clone().detach(), labels_s)
        trg_ctr_loss = center_loss(f_t, cwan_module.trg_cluster.clone().detach())
        # w-dann
        transfer_loss = domain_adv(f_s, f_t, w_s=w_s)
        domain_acc = domain_adv.domain_discriminator_accuracy
        # entropy
        entropy_loss = entropy(F.softmax(y_t, dim=1), reduction='mean')
        loss = cls_loss + src_ctr_loss * args.factor1 + trg_ctr_loss * args.factor2 \
            + transfer_loss * args.factor3 + entropy_loss * args.factor4 

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # update clusters  
        cwan_module.src_cluster_update(f_s.detach(), labels_s)
        cwan_module.trg_cluster_update(f_t.detach())


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))

    return top1.avg


def get_feature(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    feature_np = None
    target_np = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, feature = model(images)
            loss = F.cross_entropy(output, target)

            if feature_np is not None:
                feature_np = np.concatenate((feature_np, feature.cpu().numpy()), axis=0)
                target_np = np.concatenate((target_np, target.cpu().numpy()), axis=0)
            else:
                feature_np = feature.cpu().numpy()
                target_np = target.cpu().numpy()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))

    return top1.avg, feature_np, target_np


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='CWAN-V28-MixV12 w/ V18 for Partial Domain Adaptation (lambda * ent_weight * V28 + (1-lambda) * V18, MixV7 w/o entropy weights on V18)')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('--pretrain', metavar='DIR', default='', help='pretrain model path of dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--num-cluster', default=100, type=int,
                        help='Number of clusters')
    parser.add_argument('--cluster-momentum', default=0.9, type=float)
    parser.add_argument('--similarity-temp', default=1.0, type=float)
    parser.add_argument('--factor1', default=0.1, type=float,
                        help='the trade-off hyper-parameter for src center loss')
    parser.add_argument('--factor2', default=0.1, type=float,
                        help='the trade-off hyper-parameter for trg center loss')
    parser.add_argument('--factor3', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--factor4', default=0.1, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('--no-classifier-bias', default=False, action='store_true', 
                        help='whether the classifier has bias')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--mixing-factor', default=1.0, type=float)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cwanV5',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)