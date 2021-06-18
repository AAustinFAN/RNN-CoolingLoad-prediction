from torch.utils.data import Dataset, DataLoader
import train
import dataloader
import torch
from torch import nn
import numpy as np
from dataloader import mean_list,std_list


if __name__ == "__main__":

    batch_size =10
    test_size = 10000


    rnn = train.RNN_CL()
    rnn.load_state_dict(torch.load('model.pt'))
    rnn.eval()

    datax,datay = dataloader.readdata_test(mean_list,std_list)
    print(datax,datay)
    testset = train.TrainSet(datax[len(datax)-test_size:],datay[len(datax)-test_size:])
    testloader = DataLoader(testset,batch_size=batch_size)

    loss_func = nn.L1Loss()
    Loss_list = []
    for x, y in testloader:
        var_x = x.to(torch.float32)
        vay_y = y.to(torch.float32)

        prediction = rnn(var_x)
        loss = loss_func(prediction, vay_y)
        # print(loss)
        Loss_list.append(loss.item())
    print('Epoch:{}, Loss:{:.5f}'.format(1, np.mean(Loss_list)))

