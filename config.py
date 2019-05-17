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
    phase = 'train'  # "train" or "test"
    checkpoint_epoch = 30
    
    resize_or_crop = 'resize_and_crop'
    loadSize = 96  # scale image to this size
    fineSize = 96  # then crop to this size
    which_direction = 'BtoA'

    serial_batches = True
    pool_size = 0
    no_flip = True

    # optimization options
    max_epoch = 1
    batchSize = 4
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
    which_model_netG = 'unet_256'
    n_layers_D = 3  # only used if which_model_netD == n_layers
    norm = 'instance'
    no_dropout = True  # no dropout for the generator
    no_lsgan = False

    use_gpu = torch.cuda.is_available()
    gpu_ids = [0]

    lambda_A = 5
    lambda_B = 5
    lambda_identity = 0.5

    # miscs
    print_freq = 10
    save_freq = 5
    display_freq = 1
    gene_freq = 1

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