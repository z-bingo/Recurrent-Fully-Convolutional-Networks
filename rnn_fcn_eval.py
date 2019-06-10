import torch
import numpy as np
from generate_list import generate_list
from data_provider import *
from rnn_fcn_train import sampler
from torch.utils.data import DataLoader
from arch import Deep_Burst_Denoise
import torchvision.transforms as transfroms
from PIL import Image

def rnn_fcn_eval(dataset_path, txt_file, patch_size=512, model_file=None):
    # 训练模式下才会输出gt
    dataset = Data_Provider(dataset_path, txt_file, patch_size, True)
    data_loader = DataLoader(
        dataset, 1, sampler=sampler(1, dataset), num_workers=4
    )
    # load the model
    model = Deep_Burst_Denoise(1, 3).cuda()
    model.load_state_dict(torch.load(model_file))
    print('Model load OK!')
    #
    trans_tensor_rgb = transforms.ToPILImage()
    #
    model.eval()
    data_loader = iter(data_loader)
    with torch.no_grad():
        for i in range(10):
            train_data, gt = next(data_loader)
            train_data = train_data.cuda()
            gt = trans_tensor_rgb(gt.squeeze())
            gt.save('./eval_imgs/{}_gt.png'.format(i), quality=100)
            for channel in range(train_data.size(1)):
                if channel == 0:
                    sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1))
                else:
                    sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1), mfn_f)
                sfn_out = sfn_out.squeeze().detach().cpu()
                mfn_out = mfn_out.squeeze().detach().cpu()
                sfn_img = trans_tensor_rgb(sfn_out)
                mfn_img = trans_tensor_rgb(mfn_out)
                sfn_img.save('./eval_imgs/{}_sfn_{}.png'.format(i, channel), quality=100)
                mfn_img.save('./eval_imgs/{}_mfn_{}.png'.format(i, channel), quality=100)

                del sfn_out, mfn_out, sfn_img, mfn_img

                print('Save image of step {} at channel {} is OK!'.format(i, channel))

def plot_loss(path):
    import matplotlib.pyplot as plt
    loss = np.fromfile(path, np.float32, sep=',')
    plt.plot(loss)
    plt.show()

if __name__ == '__main__':
    # dataset_path = 'G:\\BinZhang\\DataSets\\rnn_fcn\\Sony\\Sony'
    # rnn_fcn_eval(
    #     dataset_path=dataset_path,
    #     txt_file='./test_list.txt',
    #     patch_size=512,
    #     model_file='./model/model_newest.pkl'
    # )
    plot_loss('./model/loss_log.txt')