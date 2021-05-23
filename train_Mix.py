import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *

parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", nargs='+', type=int, default=[3, 5, 8], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/train/Rain12600", help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--task_id", type=int)
parser.add_argument("--prefix", type=str, default='')
opt = parser.parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_id[0])) if opt.gpu_id else torch.device('cpu')

def main():

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batch_size, shuffle=True)

    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu, opt=opt)

    print_network(model)

    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.to(device)
        criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    step = 0
    tt_loss=0.0
    for epoch in range(opt.epochs):
        print("####Epoch %d Start#####" %epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        batch_num = 0
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            input_train, target_train = Variable(input_train), Variable(target_train)
            if opt.use_gpu:
                input_train, target_train = input_train.to(device), target_train.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            tt_loss += loss.data.cpu().numpy().item()

            loss.backward()
            optimizer.step()
            step += 1
            batch_num += 1
        scheduler.step(epoch)

        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch_%d.pth' % (epoch+1)))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        elif opt.data_path.find('MIXData') != -1:
            prepare_data_RainMix(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')
    main()
