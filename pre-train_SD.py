import torch.nn as nn
import torch
import sys
from load_data import load_data_1h
class StepDeep(nn.Module):
    def __init__(self):
        super(StepDeep, self).__init__()
        #input: (1,1,24,15,15)
        self.ST1=nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=16,kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(True),
            nn.Conv3d(in_channels=16, out_channels=16,kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(in_channels=16, out_channels=32,kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(True))
        self.ST2=nn.Sequential(
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(True),
            nn.Conv3d(in_channels=16, out_channels=16,kernel_size=(1, 3, 3), stride=1, padding=(0,1,1)),
            nn.ReLU(True),
            nn.Conv3d(in_channels=16, out_channels=4,kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(True)
        )
        self.fc=nn.Sequential(
            nn.Conv3d(in_channels=4,out_channels=8,kernel_size=(seq_len,1,1), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv3d(in_channels=8,out_channels=1,kernel_size=(1,1,1), stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x.transpose(1,2)
        x=self.ST1(x)
        x_rep=self.ST2(x)
        x=self.fc(x_rep)
        return x_rep,x
def adjust_learning_rate(optimizer,epoch, lr):
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
    ##########
    model = StepDeep()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    ###########开始迭代训练
    for epoch in range(iternum):
        temp_train=[]
        temp_test=[]
        ##############训练集
        for batch in range(train_X.shape[0]):  # batch数量
            # 预测结果batch*batch_size,15,15
            batch_x = train_X[batch]  # (batch_size,6,1,15,15)
            batch_y = train_Y[batch]  # (batch_size,15,15)
            rep,out = model(batch_x)
            pre_batch_y = out.squeeze()  # (batch_size,15,15)
            loss = torch.sqrt(criterion(pre_batch_y, batch_y))
            print('epoch:', epoch,'train','batch:',batch, loss.data)
            temp_train.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trainloss.append(  (sum(temp_train)/len(temp_train)) .detach().numpy() )
        ##########################################测试集
        for batch in range(test_X.shape[0]):
            batch_x = test_X[batch]  # (batch_size,6,1,15,15)
            batch_y = test_Y[batch]  # (batch_size,15,15)
            rep,out = model(batch_x)
            pre_y = out.squeeze()  # (batch_size,15,15)
            loss_test = torch.sqrt(criterion(pre_y, batch_y))
            temp_test.append(loss_test.data)
        testloss.append( (sum(temp_test)/len(temp_test)) .detach().numpy() )
    torch.save(model.state_dict(), './model/pre_trained/{}_SD.pkl'.format(source_city))
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.DoubleTensor')
    batch_size =6
    seq_len = 6
    train_size=16
    source_city='SH'
    train_X, train_Y, test_X, test_Y, label, normed_label, source_test_para, source_train_para = load_data_1h( train_size=train_size, seq_len=seq_len, city_name=source_city)
    train_source(train_X, train_Y, test_X, test_Y,source_city ,iternum=50)

