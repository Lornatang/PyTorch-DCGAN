# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generative Adversarial Networks (GANs) are one of the most interesting ideas
in computer science today. Two models are trained simultaneously by
an adversarial process. A generator ("the artist") learns to create images
that look real, while a discriminator ("the art critic") learns
to tell real images apart from fakes.
"""

import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import weights_init
from model import Generator
from model import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=True, default='./dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=256, help='inputs batch size')
parser.add_argument('--image_size', type=int, default=96, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='./checkpoints/netg_200.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='./checkpoints/netd_200.pth', help="path to netD (to continue training)")
parser.add_argument('--out_images', default='./imgs', help='folder to output images')
parser.add_argument('--out_folder', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--mode', type=str, default='train', help='model mode. default=`train`')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out_images)
    os.makedirs(opt.out_folder)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(opt.image_size),
                               transforms.CenterCrop(opt.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

fixed_noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)


def train():
    """ train model
    """
    ################################################
    #               load model
    ################################################
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        torch.load(opt.netG)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        torch.load(opt.netD)
    print(netD)

    ################################################
    #           Binary Cross Entropy
    ################################################
    criterion = nn.BCELoss()

    ################################################
    #            Use Adam optimizer
    ################################################
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    ################################################
    #               print args
    ################################################
    print("########################################")
    print(f"train dataset path: {opt.dataroot}")
    print(f"work thread: {opt.workers}")
    print(f"batch size: {opt.batch_size}")
    print(f"image size: {opt.image_size}")
    print(f"Epochs: {opt.n_epochs}")
    print(f"Noise size: {opt.nz}")
    print("########################################")
    print(f"loading pretrain model successful!\n")
    print("Starting trainning!")
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader):
            ##############################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##############################################
            # train with real
            netD.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            # real data label is 1, fake data label is 0.
            real_label = torch.full((batch_size,), 1, device=device)
            fake_label = torch.full((batch_size,), 0, device=device)

            output = netD(real_data)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ##############################################
            # (2) Update G network: maximize log(D(G(z)))
            ##############################################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
                  f"Progress->{i / len(dataloader) * 100:4.2f}% "
                  f"Loss_D: {errD.item():.4f} "
                  f"Loss_G: {errG.item():.4f} "
                  f"D(x): {D_x:.4f} "
                  f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}", end="\r")

            if i % 100 == 0:
                vutils.save_image(real_data, f"{opt.out_images}/real_samples.png", normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), f"{opt.out_images}/fake_samples_epoch_{epoch+1:03d}.png", normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), f"{opt.out_folder}/netG_epoch_{epoch + 1:03d}.pth")
        torch.save(netD.state_dict(), f"{opt.out_folder}/netD_epoch_{epoch + 1:03d}.pth")


@torch.no_grad()
def test():
    ################################################
    #               load model
    ################################################
    print(f"Load model...\n")
    netG = Generator(ngpu).eval()
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
    netG.to(device)
    print(f"Load model successful!")

    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), f"{opt.out_images}/fake.png", normalize=True)


if __name__ == '__main__':
    if opt.mode == 'train':
        train()
    elif opt.mode == 'test':
        test()
    else:
        print(opt)
