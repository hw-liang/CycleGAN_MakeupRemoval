# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
from .helper import ImagePool, GANLoss, AverageMeter

import os
import numpy as np
import torch
import scipy
import scipy.misc
from .serialization import mkdir_if_missing

class Tester(object):
    def __init__(self, opt, G_A, G_B, D_A, D_B, summary_writer):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.use_gpu and self.opt.gpu_ids else torch.device('cpu')
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B

        self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.summary_writer = summary_writer
        self.fake_B_pool = ImagePool(self.opt.pool_size)
        self.fake_A_pool = ImagePool(self.opt.pool_size)

    def test(self, epoch, data_loader):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_G = AverageMeter()
        loss_D = AverageMeter()

        start = time.time()
        for i, data in enumerate(data_loader):
            self._parse_data(data)
            data_time.update(time.time() - start)
            self._forward()

            display_dir = os.path.join(self.opt.save_dir, "%s" % self.opt.checkpoint_epoch)
            gene_dir = os.path.join(self.opt.save_dir, "%s_gene" % self.opt.checkpoint_epoch)

            AtoB = os.path.join(gene_dir,"AtoB")
            BtoA = os.path.join(gene_dir,"BtoA") 
            mkdir_if_missing(display_dir)
            mkdir_if_missing(gene_dir)
            mkdir_if_missing(AtoB)
            mkdir_if_missing(BtoA)

            if (i+1) % self.opt.display_freq == 0:
                self.sampleimages(epoch,i,display_dir)
            if (i+1) % self.opt.gene_freq == 0:
                self.geneimages(epoch,i,gene_dir)

            self.set_requires_grad([self.G_A, self.G_B, self.D_A, self.D_B], False)
            self.backward_G()
            self.backward_D_A()
            self.backward_D_B()

            batch_time.update(time.time() - start)
            start = time.time()
            loss_G.update(self.loss_G.item())
            loss_D.update(self.loss_D_A.item() + self.loss_D_B.item())
            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch {} [{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss_G {:.3f} ({:.3f})\t'
                      'Loss_D {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              loss_G.val, loss_G.mean,
                              loss_D.val, loss_D.mean))
        print('Epoch {}\tEpoch Time: {:.3f}\tLoss_G: {:.3f}\tLoss_D: {:.3f}\t'
              .format(epoch, batch_time.sum, loss_G.mean, loss_D.mean))

    def backward_D_basic(self, netD, real, fake):
        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # combine loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambdaA = self.opt.lambda_A
        lambdaB = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed
            self.idt_A = self.G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambdaB * lambda_idt
            # G_B should be identity if real_A is fed
            self.idt_B = self.G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambdaA * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A)) and D_B(G_B(B))
        self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), True)
        # forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambdaA
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambdaB
        # combine loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
                      + self.loss_idt_A + self.loss_idt_B

    def _parse_data(self, inputs):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = inputs['A' if AtoB else 'B'].to(self.device)
        self.real_B = inputs['B' if AtoB else 'A'].to(self.device)

    def _forward(self):
        self.fake_B = self.G_A(self.real_A)
        self.rec_A = self.G_B(self.fake_B)

        self.fake_A = self.G_B(self.real_B)
        self.rec_B = self.G_A(self.fake_A)

    def sampleimages(self, epoch, i, display_dir):
        A, fake_A = self.to_data(self.real_A), self.to_data(self.fake_A)
        B, fake_B = self.to_data(self.real_B), self.to_data(self.fake_B)

        merged = self.merge_images(A, fake_B)
        path = os.path.join(display_dir, 'sample-{}-{}-A-B.png'.format(epoch,i))
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))
        
        merged = self.merge_images(B, fake_A)
        path = os.path.join(display_dir, 'sample-{}-{}-B-A.png'.format(epoch,i))
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

    def geneimages(self, epoch, i, gene_dir):
        fake_A = self.to_data(self.fake_A)
        fake_B = self.to_data(self.fake_B)
        AtoB = os.path.join(gene_dir, "BtoA")
        BtoA = os.path.join(gene_dir, "AtoB")
        ### save to AtoB
        fake_B = fake_B.transpose(0,2,3,1)
        N,h,w,_ = fake_B.shape

        for j in range(N):
            temp = fake_B[j]
            path = os.path.join(AtoB, 'sample-{}-{}-A-B-{}.png'.format(epoch,i,j))
            scipy.misc.imsave(path, temp)

        ### save to BtoA
        fake_A = fake_A.transpose(0,2,3,1)
        N,h,w,_ = fake_A.shape
        for j in range(N):
            temp = fake_A[j]
            path = os.path.join(BtoA, 'sample-{}-{}-B-A-{}.png'.format(epoch,i,j))
            scipy.misc.imsave(path, temp)

    def merge_images(self, sources, targets):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.opt.batchSize))
        merged = np.zeros([3, row * h, row * w * 2])
        for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)
   
    def to_data(self,x):
        """Converts variable to numpy."""
        if self.opt.use_gpu:
            x = x.cpu()
        return x.data.numpy()
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad








