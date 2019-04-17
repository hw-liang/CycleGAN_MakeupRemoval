# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings
import torch

class DefaultConfig(object):
    seed = 0

    # dataset options
    dataroot = './datasets/makeupremoval'
    dataset_mode = 'unaligned'
    phase = 'test'  # "train" or "test"
    checkpoint_epoch = 5

    resize_or_crop = 'resize_and_crop'
    loadSize = 96  # scale image to this size
    fineSize = 96  # then crop to this size
    which_direction = 'AtoB'

    serial_batches = True
    pool_size = 0
    no_flip = True

    # optimization options
    max_epoch = 200
    batchSize = 3
    beta1 = 0.5
    lr = 2e-4  # initial learning rate adam
    lr_policy = 'lambda'
    lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
    niter = 100

    # model options
    input_nc = 3  # number of input image channels
    output_nc = 3  # number of output image channels
    ngf = 64  # number of generator filters in first conv layer
    ndf = 64  # number of discriminator filters in first conv layer
    which_model_netD = 'basic'
    which_model_netG = 'resnet_9blocks'
    n_layers_D = 3  # only used if which_model_netD == n_layers
    norm = 'instance'
    no_dropout = True  # no dropout for the generator
    no_lsgan = True

    use_gpu = torch.cuda.is_available()
    gpu_ids = [0]

    lambda_A = 0.015
    lambda_B = 0.015
    lambda_identity = 0.5

    # miscs
    print_freq = 30
    save_freq = 10
    display_freq = 10
    save_dir = './result/makeupremoval'
    workers = 10
    start_epoch = 0

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()