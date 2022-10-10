import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from load_data import load_data_1h

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size输出通道

    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # self.batch_size=batch_size
        self.padding = int((filter_size - 1) / 2)  # in this way the output has the same size

        # 4的意思是生成四个张量，i、f、o、g
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        '''

        :param input:(B,C,H,W)
        :param hidden_state: (B,C,H,W)
        :return:
        '''
        hidden, c = hidden_state  # hidden and c are images with several channels

        combined = torch.cat((input, hidden), dim=1)  # 张量连接，按照通道的维度

        A = self.conv(combined)  # (batchm,c*4,h,w)
        # print('A:',A.shape)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        # (batchm,c,h,w)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1], requires_grad=True).cpu(),
                torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1], requires_grad=True).cpu())
class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features, num_layers):
        super(CLSTM, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers

        cell_list = []
        cell_list.append(
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cpu())  # the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cpu())
        self.cell_list = nn.ModuleList(cell_list)
        self.lastconv=nn.Sequential(
            nn.Conv2d(in_channels=self.num_features,out_channels=1,kernel_size=filter_size,stride=1,padding=1),
            nn.ReLU(True)
        )
        # self.lastconv=nn.Sequential(
        #     nn.Conv2d(in_channels=self.num_features,out_channels=1,kernel_size=(1,1),stride=1,padding=0),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1), stride=1, padding=0),
        #     nn.ReLU(True)
        # )

    def forward(self, input,batch_size):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """
        hidden_state = self.init_hidden(batch_size)
        current_input = input.transpose(0, 1)  # now is seq_len,B,C,H,W
        # current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels，每一层的h,c初始化
            all_output = []
            output_inner = []  # 存放每层的所有时序的输出h
            for t in range(seq_len):  # loop for every step
                # 计算当前t的 h和c，返回（h，c）
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            # 将一层内的输出按照0维度拼接
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W
        # 返回的current_input是最终的t序列输出，next_hidden是每一层的最后时序的输出
        out=next_hidden[-1][0]#最后一层的输出h,对其卷积  (batchsize,numfeature,15,15)
        out=self.lastconv(out)
        return next_hidden, current_input,out

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples（h,c）
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
def adjust_learning_rate(optimizer,epoch, lr):
    # if epoch>15:return
    lr *= (0.1 ** (epoch // 10) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train_source(train_X, train_Y, test_X, test_Y,source_city, iternum):

    train_X = train_X.reshape(-1, batch_size,  seq_len,1,15, 15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    train_X = torch.tensor(train_X)
    train_Y = train_Y.reshape(-1, batch_size, 15, 15)
    train_Y = torch.tensor(train_Y)  # (batch,  batch_size, 15, 15)
    test_X = test_X.reshape(-1, batch_size, seq_len,1, 15, 15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    test_X = torch.tensor(test_X)
    test_Y = test_Y.reshape(-1, batch_size, 15, 15)
    test_Y = torch.tensor(test_Y)  # (batch,  batch_size, 15, 15)
    trainloss = []  # 存储训练集损失(epoch)
    testloss = []  # 存储测试集损失
    ##########模型初始化，定义损失函数,优化器
    conv_lstm = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(conv_lstm.parameters(), lr=0.0001)
    ###########开始迭代训练
    for epoch in range(iternum):
        temp_train=[]
        temp_test=[]
        ##############训练集
        for batch in range(train_X.shape[0]):  # batch数量
            # 预测结果batch*batch_size,15,15
            batch_x = train_X[batch]  # (batch_size,6,1,15,15)
            batch_y = train_Y[batch]  # (batch_size,15,15)
            next_hidden, current_input,out = conv_lstm(batch_x,batch_size)
            pre_batch_y = out.squeeze()  # (batch_size,15,15)
            #使attention相近的区域预测更接近
            loss = torch.sqrt(criterion(pre_batch_y, batch_y))

            optimizer.zero_grad()
            temp_train.append(loss.data)
            loss.backward()
            optimizer.step()
        trainloss.append(  (sum(temp_train)/len(temp_train)) .detach().numpy() )
        print('epoch:', epoch,  sum(temp_train)/len(temp_train))
    torch.save(conv_lstm.state_dict(), './model/pre_trained/{}_CL.pkl'.format(source_city))

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    ####超参数input = Variable(torch.rand(batch_size, seq_len, inp_chans, shape[0], shape[1]))
    inp_chans = 1
    num_features = 8
    filter_size = 3
    batch_size =6
    shape = (15, 15)  # H,W
    nlayers = 2
    seq_len = 6
    train_size =16
    torch.set_default_tensor_type('torch.DoubleTensor')
    source_city='SH'
    train_X, train_Y, test_X, test_Y, label, normed_label, test_para, train_para =load_data_1h(train_size=train_size, seq_len=seq_len, city_name=source_city)
    train_source(train_X, train_Y, test_X, test_Y,source_city ,iternum=50)

