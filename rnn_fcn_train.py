import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import argparse
import rawpy
from data_provider import *
from arch import Deep_Burst_Denoise
import torch.nn.functional as F
import torch.optim as optim
from generate_list import generate_list
from PIL import Image

TENSOR_BOARD = False
if TENSOR_BOARD:
    from tensorboardX import SummaryWriter
"""
The dataloader sampler, to ensure the size of each batch are same.
"""
class sampler(Sampler):
    def __init__(self, batch_size, data_source):
        super(sampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.total_size = len(data_source)

    def __iter__(self):
        if self.total_size % self.batch_size == 0:
            return iter(torch.randperm(self.total_size))
        else:
            return iter(torch.randperm(self.total_size).tolist() + torch.randperm(self.total_size).tolist()[:self.batch_size-self.total_size % self.batch_size])

    def __len__(self):
        return self.total_size


def adjust_learning_rate(optimizer, lr):
    """Decay the learning rate by one half"""
    lr = lr / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def rnn_fcn_train(dataset_path, txt_file, batch_size=4, patch_size=512, lr=5e-4, lr_decay=1000, max_epoch=10000):
    dataset = Data_Provider(dataset_path, txt_file, patch_size=patch_size, train=True)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler(batch_size, dataset),
        num_workers=4
    )
    # build the architecture
    model = Deep_Burst_Denoise(1, 3).cuda()
    # switch to the train mode
    model.train()
    # init the optimizer to train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # if tensorboardX is used, saving the loss logs
    if TENSOR_BOARD:
        summary = SummaryWriter('./logs', comment='loss')
    # create a txt file to save loss
    loss_file = open('./loss_logs/loss_log.txt', 'w+')
    # the variable to save loss during the train process
    global_step = 0
    # to ensure the losses are saved correctly, initial a max value to min_loss
    min_loss = 10**9+7
    try:
        for epoch in range(max_epoch):
            # learning rate decay
            if epoch > 0 and epoch % lr_decay == 0:
                lr = adjust_learning_rate(optimizer, lr)
            print('=============Epoch:{}, lr:{}.============'.format(epoch+1, lr))
            for step, (train_data, gt) in enumerate(data_loader):
                train_data = train_data.cuda()
                gt = gt.cuda()
                loss_temp = 0
                for channel in range(train_data.size(1)):
                    if channel == 0:
                        sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1))
                    else:
                        sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1), mfn_f)
                    # 计算loss
                    loss_temp += F.l1_loss(sfn_out, gt) + F.l1_loss(mfn_out, gt)
                # 保存loss到Tensorboard
                global_step += 1
                if TENSOR_BOARD:
                    summary.add_scalar('loss', loss_temp, global_step)
                # 打印信息
                print('Epoch:{}, Step:{}, Loss:{:.4f}.'.format(epoch+1, step, loss_temp))
                # 优化
                optimizer.zero_grad()
                loss_temp.backward()
                optimizer.step()
                # save some temp images
                # if global_step % 20 == 0:
                #     img = mfn_out.detach()
                #     summary.add_image('image', img.squeeze(0), global_step)

                # save loss_temp to file
                loss_file.write('{},'.format(loss_temp))
                # save the model
                if loss_temp < min_loss:
                    min_loss = loss_temp
                    torch.save(model.state_dict(), 'model_min_loss.pkl')
                if global_step % 1000 == 0:
                    torch.save(model.state_dict(), 'model_newest.pkl')
    finally:
        loss_file.close()

if __name__ == '__main__':
    dataset_path = 'G:\\BinZhang\\DataSets\\rnn_fcn\\Sony\\Sony'
    generate_list(dataset_path, './', burst=8)
    rnn_fcn_train(
        dataset_path=dataset_path,
        txt_file='./train_list.txt',
        batch_size=1,
        patch_size=512,
        lr=5e-5
    )
