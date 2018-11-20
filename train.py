from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from resnet import resnet50
from yoloLoss import YoloLoss
from dataset import Yolodata
import numpy as np
#from visualize import Visualizer
import numpy as np
import pandas as pd

import time

parser = argparse.ArgumentParser(description='PyTorch Yolov1 Training DuanYiqun')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--experimentname',default='sdmnv5_vscommon', type=str, help='model name for save')
parser.add_argument('--trainroot',default='/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', type=str, help='for train images')
parser.add_argument('--testroot',default='/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', type=str, help='for test images')
parser.add_argument('--indexdir',default='./data/yolo_train_on_VOC2012.csv', type=str, help='log direction for save')
parser.add_argument('--batch_size',default=20, type=int, help='batch size')
parser.add_argument('--num_epochs',default=50, type=int, help='training length')
parser.add_argument('--sgrid',default=7, type=int, help='grid number 7*7 for default')
parser.add_argument('--bbxnumber',default=2, type=int, help='bounding box number')
parser.add_argument('--classnumber',default=20, type=int, help='class number default is 20')
parser.add_argument('--mname',default='train50', type=str, help='experimentname')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_loss = 1000000  
# best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('loading dataset ...')
Datasetinstance = Yolodata(train_file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', train_listano = './voc2012.txt', test_file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', test_listano = './voc2012.txt' ,batchsize=args.batch_size, snumber = args.sgrid, bnumber = args.bbxnumber, cnumber = args.classnumber)
train_loader, test_loader = Datasetinstance.getdata()

print('the dataset has %d images for train' % (len(train_loader)))
print('the batch_size is %d' % (args.batch_size))

print('loading network structure ...')
net = resnet50()
net = net.to(device)
print(net) 

print('load pre-trined model')
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
op = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in op.keys() and not k.startswith('fc'):
        op[k] = new_state_dict[k]
net.load_state_dict(op)

if args.resume:
    net.load_state_dict(torch.load('best.plk'))

print( 'testing the cuda device here')
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = YoloLoss(args.batch_size,args.bbxnumber,args.classnumber,lambda_coord= 0.5, lambda_noobj = 0.5)

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':args.lr*1}]
    else:
        params += [{'params':[value],'lr':args.lr}]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print('batch %s of total batch %s' % (batch_idx, len(train_loader)), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))

    end_time=time.time()
    epoch_time=end_time-start_time
    data=[epoch,train_loss/(batch_idx+1),epoch_time]
    print('trainloss:{},time_used:{}'.format(train_loss/(batch_idx+1),epoch_time))
    return data

def test(epoch):
    global best_loss
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0

    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        pred = net(inputs)
        loss = criterion(pred,targets)

        test_loss += loss.item()

        print('batch %s of total batch %s' % (batch_idx, len(test_loader)), 'Loss: %.3f ' % (test_loss/(batch_idx+1)))

    end_time=time.time()
    epoch_time=end_time-start_time
    data=[epoch,test_loss/(batch_idx+1),epoch_time]
    print('trainloss:{},time_used:{}'.format(test_loss/(batch_idx+1),epoch_time))

    if test_loss < best_loss:
        print('Updating best validation loss')
        best_loss = test_loss
        print('Saving..best_record')
        state = {
            'net': net.state_dict(),
            'loss': best_loss,
            'epoch': epoch,
        }

        savepath='./train/'+str(args.mname)+'/checkpoints'
        filename = savepath + '/best_check.plk'
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        torch.save(state, filename)

    return data

a=[1,2,3]
trainnp=np.array(a)
testnp=np.array(a)

nums=[]
sparses=[]

sparsity_book=[]

def extract(m):
    global sparses
    global nums
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nums.append(torch.numel(m.weight.data))
        cc=m.weight.clone().cpu()
        sparses.append(torch.mean(cc.abs()).detach().numpy())

for epoch in range(args.num_epochs):
    learning_rate = args.lr
    if epoch == 30:
        learning_rate=0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    if epoch == 45:
        learning_rate=0.0001
    #optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            
    print('\n\nStarting epoch %d / %d' % (epoch + 1, args.num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    nd = train(epoch)
    trainnp=np.vstack((trainnp,np.array(nd)))
    ed = test(epoch)
    testnp=np.vstack((testnp,np.array(ed)))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]
    savepath='./train/'+str(args.mname)+'train.csv'
    train_data=pd.DataFrame(trainnp,columns=['epoch','loss','epoch_time'])
    train_data.to_csv(savepath)
    savepath='./train/'+str(args.mname)+'test.csv'
    test_data=pd.DataFrame(testnp,columns=['epoch','loss','epoch_time'])
    test_data.to_csv(savepath)