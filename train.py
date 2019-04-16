# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import sys
from pprint import pprint
import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config import opt
from utils.serialization import Logger, save_checkpoint
from datasets.data_provider import UnalignedDataset
from models.networks import define_D, define_G
from utils.trainer import Trainer

def train_cycle_gan(**kwargs):
    opt._parse(kwargs)
    torch.manual_seed(opt.seed)

    # Write standard output into file
    sys.stdout = Logger(os.path.join(opt.save_dir, 'log_train.txt'))

    print('========user config========')
    pprint(opt._state_dict())
    print('===========end=============')
    if opt.use_gpu:
        print('currently using GPU')
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    pin_memory = True if opt.use_gpu else False

    print('initializing dataset {}'.format(opt.dataset_mode))
    dataset = UnalignedDataset(opt)
    trainloader = DataLoader(dataset, opt.batchSize, True, num_workers=opt.workers, pin_memory=pin_memory)

    summaryWriter = SummaryWriter(os.path.join(opt.save_dir, 'tensorboard_log'))

    print('initializing model ... ')
    use_dropout = not opt.no_dropout
    netG_A = define_G(opt.input_nc, opt.output_nc, opt.ndf, opt.which_model_netG, opt.norm, use_dropout)
    netG_B = define_G(opt.output_nc, opt.input_nc, opt.ndf, opt.which_model_netG, opt.norm, use_dropout)
    use_sigmoid = opt.no_lsgan
    netD_A = define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid)
    netD_B = define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid)

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()),
                                lr=opt.lr, betas=(opt.beta1, 0.999))

    def get_scheduler(optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.start_epoch - opt.niter) / float(opt.lr_decay_iters + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [{}] is not implemented'.format(opt.lr_policy))
        return scheduler

    scheduler_G = get_scheduler(optimizer_G, opt)
    scheduler_D = get_scheduler(optimizer_D, opt)

    start_epoch = opt.start_epoch
    if opt.use_gpu:
        netG_A = torch.nn.DataParallel(netG_A).cuda()
        netG_B = torch.nn.DataParallel(netG_B).cuda()
        netD_A = torch.nn.DataParallel(netD_A).cuda()
        netD_B = torch.nn.DataParallel(netD_B).cuda()

    # get trainer
    cycleganTrainer = Trainer(opt, netG_A, netG_B, netD_A, netD_B, optimizer_G, optimizer_D, summaryWriter)
    # start training
    for epoch in range(start_epoch, opt.max_epoch):
        scheduler_G.step()
        scheduler_D.step()
        # train over whole dataset
        cycleganTrainer.train(epoch, trainloader)
        if (epoch + 1) % opt.save_freq == 0 or (epoch + 1) == opt.max_epoch:
            if opt.use_gpu:
                state_dict_netG_A = netG_A.module.state_dict()
                state_dict_netG_B = netG_B.module.state_dict()
                state_dict_netD_A = netD_A.module.state_dict()
                state_dict_netD_B = netD_B.module.state_dict()
            else:
                state_dict_netG_A = netG_A.state_dict()
                state_dict_netG_B = netG_B.state_dict()
                state_dict_netD_A = netD_A.state_dict()
                state_dict_netD_B = netD_B.state_dict()
            save_checkpoint({
                'netG_A': state_dict_netG_A,
                'netG_B': state_dict_netG_B,
                'netD_A': state_dict_netD_A,
                'netD_B': state_dict_netD_B,
                'epoch': epoch + 1,
            }, False, save_dir=opt.save_dir, filename='checkpoint_ep' + str(epoch + 1))





