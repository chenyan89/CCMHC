import torch
import  numpy as np
import torch.nn as nn
from load_data import load_data_1h
import matplotlib.pyplot as plt
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
        x_rep1=self.ST1(x)
        x_rep2=self.ST2(x_rep1)
        x=self.fc(x_rep2)
        return x_rep1,x_rep2,x
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
    match_loss=sum(region_loss)/len(region_loss)
    return match_loss
def adjust_learning_rate(optimizer,epoch, lr):
    lr *= (0.1 ** (epoch // 10) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train(XS,XT,YT,iter_num,source_city,target_city,w):

    #################模型加载,优化器加载
    G =StepDeep()
    G.load_state_dict(torch.load('./model/pre_trained/{}_SD.pkl'.format(source_city)))
    G_source=StepDeep()
    G_source.load_state_dict(torch.load('./model/pre_trained/{}_SD.pkl'.format(source_city)))
    for p in G.ST1.parameters():
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
        batch_test_pre = []
        batch_match=[]
        #set batch_size=1,we want to get the time_slot of each input
        for batch in range(XS.shape[0]):  # batch数量，都是5天的，1h跨度，seq=6 XS.shape[0]
            batch_XS = XS[batch]  # (batch_size,6,1,15,15)
            batch_XT = XT[batch]  # (batch_size,6,1,15,15)
            batch_YT = YT[batch]  # (batch_size,15,15)
            _s, rep_s, outs = G_source(batch_XS)
            _t, rep_t, outt = G(batch_XT)
            pre_target = outt.squeeze()  # (batch_size,15,15)
            time_slot=np.array([0,1,2,3,4,5])+np.array([(batch%24)*6]*6) %24  ##当前输入处于一天的时间

            ####################################计算loss##########################################
            loss_match=transfer_loss(rep_s,rep_t,attention=attention,time_slot=time_slot)
            loss_pre=get_pre_loss(pre_target, batch_YT)
            loss=w*loss_match+(1-w)*loss_pre
            ######
            batch_match.append(loss_match.detach().numpy())
            batch_train_pre.append(loss_pre.detach().numpy())
            #更新参数
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
        train_loss_pre.append(np.mean(batch_train_pre))
        print(epoch, 'loss_pre:', np.mean(batch_train_pre), 'loss_match', np.mean(batch_match))
    torch.save(G.state_dict(),  './model/{}-{}/CCMHC_SD_mutual.pkl'.format(source_city,target_city))

def get_pre(test_X,target_label,para,source_city,target_city):

    ###################################################验证模型
    model = StepDeep()
    model.load_state_dict(torch.load('./model/{}-{}/CCMHC_SD_mutual.pkl'.format(source_city,target_city)))
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
    print('rmse,mape of {}-{} CCMHC_SD:'.format(source_city, target_city), rmse,mape)
    # np.save('./result/{}-{}_CCMHC_SD.npy'.format(source_city,target_city), rever_pre)

def data_preprocessing(XS,YS,source_test_x,source_test_y,XT,YT,test_X,test_Y):
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
    ## choose source_city and target_city: SH,HK,NJ
    source_city='NJ'
    target_city='HK'
    ######################
    batch_size =6
    seq_len = 6
    train_size=5
    ###### load data
    torch.set_default_tensor_type('torch.DoubleTensor')
    train_X, train_Y, test_X, test_Y, label, normed_label, source_test_para,source_train_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=source_city)
    target_train_X, target_train_Y, target_test_X, target_test_Y, target_label, target_normed_label, target_para,train_target_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=target_city)
    ### npy to tensor
    train_X,train_Y, test_X, test_Y, target_train_X,target_train_Y,target_test_X,target_test_Y\
        =data_preprocessing(train_X,train_Y,test_X, test_Y,target_train_X,target_train_Y,target_test_X,target_test_Y)
    ## train
    train(train_X,target_train_X,target_train_Y,50,source_city,target_city,w=0.6)
    ## test
    get_pre(target_test_X,target_normed_label,target_para,source_city,target_city)

