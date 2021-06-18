import dataloader
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

input_dim = 2
hidden_size = 128
num_layers = 1
batch_size = 10
train_ratio = 0.7
learningRate = 0.02
epoch = 10

class RNN_CL(nn.Module):
    def __init__(self):
        super(RNN_CL, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,batch_first=True)  # rnn 2 6 2
        # rnn = nn.LSTM(inp_dim, mid_dim, num_layers)
        # # inp_dim 是LSTM输入张量的维度，我们已经根据我们的数据确定了这个值是2
        # # mid_dim 是LSTM三个门 (gate) 的网络宽度，也是LSTM输出张量的维度
        # # num_layers 是使用两个LSTM对数据进行预测，然后将他们的输出堆叠起来。
        self.out = nn.Sequential(nn.Linear(hidden_size, 1))  # 回归

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全0的 state
        batch_size,seq,hidden_size = r_out.shape
        out = self.out(h_n.view(batch_size,-1))
        return out

class TrainSet(Dataset):
    def __init__(self, datax,datay):
        # 定义好 image 的路径
        self.data, self.label = datax, datay

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def main():
    print('start to train')

    rnn = RNN_CL()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learningRate)
    loss_func = nn.L1Loss()

    for step in range(epoch):
        Loss_list = []
        prelist = []

        for x,y in trainloader:
            var_x= x.to(torch.float32)
            vay_y= y.to(torch.float32)

            prediction = rnn(var_x)
            for x in prediction:
                prelist.append(x.detach().numpy())
            print(prediction)
            print('----')
            loss = loss_func(prediction,vay_y)

            Loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:{}, Loss:{:.5f}'.format(step + 1, np.mean(Loss_list)))

    # x=  np.linspace(0,10000 ,10000)
    # plt.plot(x,prelist)
    # plt.plot(x,trainset.label)
    # # plt.show()

    torch.save(rnn.state_dict(),'model.pt')

if __name__ == "__main__":


    datax, datay,= dataloader.readdata_train()
    train_size = round(len(datax) * train_ratio) # get a integer train_size

    print('data size is ', datax.shape, datay.shape)

    trainset = TrainSet(datax[:train_size], datay[:train_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    main()