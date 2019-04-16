# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os
import shutil
import sys
import torch
from models.networks import define_D, define_G

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath = None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')
    def __del__(self):
        self.close()
    def __enter__(self):
        pass
    def __exit__(self, *args):
        self.close()
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    fpath = '_'.join((str(state['epoch']), filename))
    fpath = os.path.join(save_dir, fpath)
    mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(save_dir, 'model_best.pth.tar'))

def load_checkpoint(opt):
    """Loads the generator and discriminator models from checkpoints.
    """
    use_dropout = not opt.no_dropout
    netG_A = define_G(opt.input_nc, opt.output_nc, opt.ndf, opt.which_model_netG, opt.norm, use_dropout)
    netG_B = define_G(opt.output_nc, opt.input_nc, opt.ndf, opt.which_model_netG, opt.norm, use_dropout)
    use_sigmoid = opt.no_lsgan
    netD_A = define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid)
    netD_B = define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid)

    checkpoint_file = "%d_checkpoint_ep%d" % (opt.checkpoint_epoch, opt.checkpoint_epoch)
    checkpoint = torch.load(os.path.join(opt.save_dir,checkpoint_file))

    netG_A.load_state_dict(checkpoint['netG_A'])
    netG_B.load_state_dict(checkpoint['netG_B'])
    netD_A.load_state_dict(checkpoint['netD_A'])
    netD_B.load_state_dict(checkpoint['netD_B'])
    # start_epoch = checkpoint['epoch'] + 1

    return netG_A, netG_B, netD_A, netD_B