# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from pprint import pprint
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import opt
from utils.serialization import Logger, load_checkpoint
from datasets.data_provider import UnalignedDataset
from utils.tester import Tester

def test_cycle_gan(**kwargs):
    opt._parse(kwargs)
    torch.manual_seed(opt.seed)

    # Write standard output into file
    sys.stdout = Logger(os.path.join(opt.save_dir, 'log_test.txt'))

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
    testloader = DataLoader(dataset, opt.batchSize, True, num_workers=opt.workers, pin_memory=pin_memory)

    summaryWriter = SummaryWriter(os.path.join(opt.save_dir, 'tensorboard_log'))

    print('initializing model ... ')
    netG_A, netG_B, netD_A, netD_B = load_checkpoint(opt)
    start_epoch = opt.start_epoch
    if opt.use_gpu:
        netG_A = torch.nn.DataParallel(netG_A).cuda()
        netG_B = torch.nn.DataParallel(netG_B).cuda()
        netD_A = torch.nn.DataParallel(netD_A).cuda()
        netD_B = torch.nn.DataParallel(netD_B).cuda()

    # get tester
    cycleganTester = Tester(opt, netG_A, netG_B, netD_A, netD_B, summaryWriter)

    for epoch in range(start_epoch, opt.max_epoch):
        # test over whole dataset
        cycleganTester.test(epoch, testloader)

if __name__ == '__main__':
    test_cycle_gan()