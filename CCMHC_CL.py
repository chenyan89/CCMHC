import torch.nn as nn
import torch
import numpy as np

from load_data import load_data_1h

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
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
        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=1, kernel_size=filter_size, stride=1, padding=1),
            nn.ReLU(True)
        )

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
    lr *= (0.1 ** (epoch // 10) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def transfer_loss(region1,region2,attention,time_slot):
    '''
    :param region1: （6,4,6,15,15)
    :param region2:
    :param attention:  流量分时段相关性(24,225,2)  [regionidx,corr]
    time_slot:  size:6;  0-23 确定属于哪一个时间段  [0,1,2,3,4,5]-[18,19,20,21,22,23]
    :return:
    '''
    region_loss=[]
    region1=region1.reshape(batch_size,-1,225)  ##(6,feature,region_idx) (6,24,225)
    region2=region2.reshape(batch_size,-1,225)
    # which_slot=[0]*7+[1]*3+[2]*4+[3]*5+[4]*5 ##时段定位5
    which_slot=list(range(24))# 24
    for tidx in range(attention.shape[1]): ###每个目标区域进行迁移
        if target_para[tidx, 1] == 0.0:continue     #####空区域
        batch_loss=[]
        for i in range(time_slot.shape[0]):
            #每个batch的时段
            slot_idx=which_slot[time_slot[i]]
            match_ridx = int(attention[slot_idx, tidx, 0])
            match_corr = attention[slot_idx, tidx, 1]
            if match_corr<0.1:
                continue
            else:
                # print(slot_idx)
                s_rep = region1[i,:, match_ridx]
                t_rep = region2[i,:,tidx]

                batch_loss.append(match_corr * torch.mean(torch.pow(s_rep - t_rep, 2))) #RMSE
        if batch_loss:
            region_loss.append(sum(batch_loss)/len(batch_loss))
    match_loss=sum(region_loss)
    return match_loss
def get_pre_loss(pre,label):
    '''
    :param pre:  (batch_size,15,15)
    :param label:  (batch_size,15,15)
    :return: loss
    '''
    sum_loss=[]
    for batch in range(label.shape[0]):
        b=pre[batch]
        c=label[batch]
        loss=torch.sum(  torch.pow(b-c,2)  )
        sum_loss.append(loss)
    loss=sum(sum_loss)/len(sum_loss)
    return loss

def train(XS,XT,YT,iter_num,source_city,target_city,w):
    #################模型加载,优化器加载
    G = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    G_source = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    G.load_state_dict(torch.load( './model/pre_trained/{}_CL.pkl'.format(source_city)))
    G_source.load_state_dict(torch.load( './model/pre_trained/{}_CL.pkl'.format(source_city)))
    for p in G.cell_list[0].parameters():
        p.requires_grad = False
    for p in G_source.parameters():
        p.requires_grad = False
    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,G.parameters()), lr=0.01)

    ############开始迭代训练
    # attention=np.load('./{}-{}_filter.npy'.format(source_city,target_city)) #size=(time_slot,225,2)
    attention=np.load('./{}-{}_filter_mutual.npy'.format(source_city,target_city)) #size=(time_slot,225,2)
    train_loss_pre=[]
    for epoch in range(iter_num):
        adjust_learning_rate(G_optimizer, epoch, 0.01)
        batch_train_pre=[]
        batch_match=[]
        for batch in range(XS.shape[0]):  # batch数量，都是5天的，1h跨度，seq=6 XS.shape[0]
            batch_XS = XS[batch]  # (batch_size,6,1,15,15)
            batch_XT = XT[batch]  # (batch_size,6,1,15,15)
            batch_YT = YT[batch]  # (batch_size,15,15)
            _s, rep_s, outs = G_source(batch_XS,batch_size)
            _t, rep_t, outt = G(batch_XT,batch_size)
            rep_s=rep_s[:,-1,:,:,:]
            rep_t=rep_t[:,-1,:,:,:]
            #############优化参数
            pre_target = outt.squeeze()  # (batch_size,15,15)
            time_slot = np.array([0, 1, 2, 3, 4, 5]) + np.array([(batch % 24) * 6] * 6) % 24
            ####################################计算loss##########################################
            loss_match = transfer_loss(rep_s, rep_t, attention=attention, time_slot=time_slot)
            loss_pre = get_pre_loss(pre_target, batch_YT)
            loss = w * loss_match + (1-w)* loss_pre
            ######
            batch_match.append(loss_match.detach().numpy())
            batch_train_pre.append(loss_pre.detach().numpy())
            # 更新参数
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
        train_loss_pre.append(np.mean(batch_train_pre))
        print(epoch, 'loss_pre:', np.mean(batch_train_pre), 'loss_match', np.mean(batch_match))
    torch.save(G.state_dict(),  './model/{}-{}/CCMHC_CL_mutual.pkl'.format(source_city,target_city))

################################# result
def get_pre(test_X,target_label,para,source_city,target_city):

    ###################################################验证模型
    model = CLSTM(shape,inp_chans,filter_size,num_features,nlayers)

    model.load_state_dict(torch.load('./model/{}-{}/CCMHC_CL_mutual.pkl'.format(source_city,target_city)))
    for param in model.parameters():
        param.requires_grad = False
    average_pre = []  # (batch,batch_size，15，15)
    for batch in range(test_X.shape[0]):
        batch_x = test_X[batch]  # (batch_size,6,1,15,15)
        _,rep,out = model(batch_x,batch_size)  # 返回(next_hidden, current_input)
        pre_y = out.squeeze()  # (batch_size,15,15)
        average_pre.append(pre_y.data)
    average_pre = np.array([item.detach().numpy() for item in average_pre])
    average_pre = average_pre.reshape(-1, 225)
    count=0
    ###将空区域赋0，预测值为负赋0
    for re_id in range(225):
        if para[re_id][1]==0:
            count+=1
            average_pre[:, re_id]=average_pre[:, re_id]*0
        else:
            temp=average_pre[:,re_id]
            for i in range(len(temp)):
                if temp[i]<0:
                    average_pre[i,re_id]=0
    ## Denormalize
    para_min=para[:,0]
    para_max=para[:,1]
    rever_pre=[]#反归一化的预测值
    for i in range(225):
        c = (average_pre[:, i] * (para_max[i] - para_min[i])) + para_min[i]
        for idx in range(len(c)):
            if c[idx]<0:
                c[idx]=0
        rever_pre.append(c)
    rever_pre=np.asarray(rever_pre)  #(225, 330)
    rever_pre=np.swapaxes(rever_pre,0,1)#(330, 225)

    rmse = my_RMSE(rever_pre, target_label)/(225-count)
    # MAE=my_MAE(rever_pre, target_label)/(225-count)
    mape=my_MAPE(rever_pre, target_label)/(225-count)
    print('rmse,mape of {}-{} CCMHC_CL:'.format(source_city, target_city), rmse,mape)
    # np.save('./result/{}-{}_CCMHC_CL.npy'.format(source_city,target_city), rever_pre)
def data_preprocessing(XS,YS,source_test_x,source_test_y,XT,YT,test_X,test_Y):

    #####数据处理
    XS = XS.reshape(-1, batch_size,  seq_len,1,15, 15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    XS = torch.tensor(XS)
    YS = YS.reshape(-1, batch_size, 15, 15)
    YS = torch.tensor(YS)  # (batch,  batch_size, 15, 15)
    XT= XT.reshape(-1, batch_size,  seq_len,1,15, 15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    XT = torch.tensor(XT)
    YT = YT.reshape(-1, batch_size, 15, 15)
    YT = torch.tensor(YT)  # (batch,  batch_size, 15, 15)
    test_X = test_X.reshape(-1, batch_size, seq_len,1, 15, 15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    test_X = torch.tensor(test_X)
    test_Y = test_Y.reshape(-1, batch_size, 15, 15)
    test_Y = torch.tensor(test_Y)  # (batch,  batch_size, 15, 15)

    source_test_x = source_test_x.reshape(-1, batch_size, seq_len,1, 15, 15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    source_test_x = torch.tensor(source_test_x)
    source_test_y = source_test_y.reshape(-1, batch_size, 15, 15)
    source_test_y = torch.tensor(source_test_y)  # (batch,  batch_size, 15, 15)
    return XS,YS,source_test_x,source_test_y,XT,YT,test_X,test_Y
def my_RMSE(data,label):
    return sum(np.sqrt(np.mean(pow(data - label, 2),axis=0)))
def my_MAE(data,label):
    return sum(np.mean(np.abs(data-label),axis=0) )
def my_MAPE(data,label):
    final_loss=np.zeros(225)
    for i in range(label.shape[1]):
        #region
        time_num=0
        gap=0
        for j in range(label.shape[0]):
        #time
            if  label[j][i]<10:
                data[j][i]=0
            else:
                time_num+=1
                gap+=np.abs(data[j][i]-label[j][i])/label[j][i]
        if time_num!=0:
            final_loss[i]=gap/time_num
        else:
            final_loss[i]=0
    return sum(final_loss)
if __name__ == '__main__':

    ## parameters
    source_city='SH'
    target_city='NJ'
    inp_chans = 1
    num_features = 8
    filter_size = 3
    batch_size =6
    shape = (15, 15)
    nlayers = 2
    seq_len = 6
    train_size = 5
    ############ load data
    torch.set_default_tensor_type('torch.DoubleTensor')
    train_X, train_Y, test_X, test_Y, label, normed_label, source_test_para,source_train_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=source_city)
    target_train_X, target_train_Y, target_test_X, target_test_Y, target_label, target_normed_label, target_para,train_target_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=target_city)
    ### numpy to tensor
    train_X,train_Y, test_X, test_Y, target_train_X,target_train_Y,target_test_X,target_test_Y\
        =data_preprocessing(train_X,train_Y,test_X, test_Y,target_train_X,target_train_Y,target_test_X,target_test_Y)

    #### train
    train(train_X,target_train_X,target_train_Y,50,source_city,target_city,w=0.6)

    #### test
    get_pre(target_test_X, target_label,target_para,source_city,target_city)

