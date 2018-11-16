#encoding:utf-8
#
#created by xiongzihua
#
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class VOCDataset(data.Dataset):
    image_size = 448
    def __init__(self,root,list_file,train,transform,loader):
        print('loading annotations')
        self.loader = loader
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = self.loader(os.path.join(self.root+fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.transform is not None:
            img = self.transform(img)

        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        target = self.encoder(boxes,labels)# 7x7x30

        return img,target

    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        grid_num = 14
        target = torch.zeros((grid_num,grid_num,30))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

class Yolodata():
    def __init__(self, file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', listano = './voc2012.txt',batchsize=2):
        transform_train = transforms.Compose([
                       transforms.Resize(224),
                       transforms.RandomCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        img_data = VOCDataset(root = file_root,list_file=listano,train=True,transform=transform_train,loader = pil_loader)
        train_loader = torch.utils.data.DataLoader(img_data, batch_size=batchsize,shuffle=True)
        self.train_loader = train_loader
        #self.img_data=img_data

        transform_test = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        img_data_t = VOCDataset(root = file_root,list_file='voc2012.txt',train=False,transform=transform_test,loader = pil_loader)
        test_loader = torch.utils.data.DataLoader(img_data_t, batch_size=int(0.5*batchsize),shuffle=False)
        self.test_loader = test_loader
    
    def test(self):
        #print(len(self.img_data))
        print('there are total %s batches in training and total %s batches for test' % (len(self.train_loader),len(self.test_loader)))
        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            print( batch_x.size(), batch_y)
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            print( batch_x.size(), batch_y)
    
    def getdata(self):
        return self.train_loader, self.test_loader


if __name__ == '__main__':
    testdata = Yolodata(file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', listano = './voc2012.txt',batchsize=2)
    testdata.test()


