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
import cv2

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

def cv_loader(path):
    #opencv
    return cv2.imread(path)

class VOCDataset(data.Dataset):
    def __init__(self,root,list_file,train,transform,loader,snumber = 7,bnumber = 2,cnumber =20, image_size = 448):
        print('loading annotations')
        self.loader = loader
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.S = snumber # grid number 7*7 normally
        self.B = bnumber # bounding box number in each grid
        self.C = cnumber # how many classes
        self.mean = (123,117,104)#RGB
        self.image_size =image_size

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
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = self.loader(os.path.join(self.root+fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        img,boxes,labels = self.cvtransform(img,boxes,labels) # 各种变换用torch 自带的的transform做不到，所以借鉴了xiongzihua 的git hub上的代码写了一点cv变换
        h,w,_= img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        #print('bounding box is ')
        ##print(labels)
        #target = self.encoder(boxes,labels)# 7x7x30
        target = self.make_target(labels,boxes)
        target = torch.tensor(target).float()

        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        #img = self.subMean(img,self.mean) #减去均值
        img = cv2.resize(img,(self.image_size,self.image_size))

        if self.transform is not None:
            img = self.transform(img)
        print(img)
        
        return img,target
    
    def cvtransform(self, img, boxes, labels):
        img, boxes = self.random_flip(img, boxes)
        img,boxes = self.randomScale(img,boxes)
        img = self.randomBlur(img)
        img = self.RandomBrightness(img)
        img = self.RandomHue(img)
        img = self.RandomSaturation(img)
        img,boxes,labels = self.randomShift(img,boxes,labels)
        img,boxes,labels = self.randomCrop(img,boxes,labels)
        return img,boxes,labels



    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return [self.S, self.S, self.B*5+self.C]
        '''
        grid_num = 14
        target = torch.zeros((grid_num,grid_num,30))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2] #这是x-x2和 y-y2
        cxcy = (boxes[:,2:]+boxes[:,:2])/2 #这时中心点坐标
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
    
    def change_box_to_center_axes(self, bboxes):
        rebboxes = []
        for bbox in bboxes:
            x_center, y_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            width, height = (bbox[0] - bbox[2]), (bbox[1] - bbox[3])
            rebboxes.append([x_center, y_center, width, height])
        return rebboxes

    def make_target(self, labels, bboxes):
        """make location np.ndarray from bboxes of an image
        
        Parameters
        ----------
        labels : list
            [0, 1, 4, 2, ...]
            labels of each bboxes
        bboxes : list
            [[x_center, y_center, width, height], ...]
        
        Returns
        -------
        np.ndarray
            [self.S, self.S, self.B*5+self.C]
            location array
        """

        bboxes = self.change_box_to_center_axes(bboxes)
        num_elements = self.B*5 + self.C
        num_bboxes = len(bboxes)
        
        # for excetion: num of bboxes is zero
        if num_bboxes == 0:
            return np.zeros((self.S, self.S, num_elements))

        labels = np.array(labels, dtype=np.int)
        bboxes = np.array(bboxes, dtype=np.float)

        np_target = np.zeros((self.S, self.S, num_elements))
        np_class = np.zeros((num_bboxes, self.C))

        for i in range(num_bboxes):
            np_class[i, labels[i]] = 1

        x_center = bboxes[:, 0].reshape(-1, 1)
        y_center = bboxes[:, 1].reshape(-1, 1)
        w = bboxes[:, 2].reshape(-1, 1)
        h = bboxes[:, 3].reshape(-1, 1)

        x_idx = np.ceil(x_center * self.S) - 1
        y_idx = np.ceil(y_center * self.S) - 1
        # for exception 0, ceil(0)-1 = -1
        x_idx[x_idx<0] = 0
        y_idx[y_idx<0] = 0

        # calc offset of x_center, y_center
        x_center = x_center - x_idx/self.S - 1/(2*self.S)
        y_center = y_center - y_idx/self.S - 1/(2*self.S)

        conf = np.ones_like(x_center)

        temp = np.concatenate([x_center, y_center, w, h, conf], axis=1)
        temp = np.repeat(temp, self.B, axis=0).reshape(num_bboxes, -1)
        temp = np.concatenate([temp, np_class], axis=1)

        for i in range(num_bboxes):
            np_target[int(y_idx[i]), int(x_idx[i])] = temp[i]

        return np_target
    

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels
    
    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels
    
    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

class Yolodata():
    def __init__(self, file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', listano = './voc2012.txt',batchsize=2):
        transform_train = transforms.Compose([
                       #transforms.Resize(448),
                       #transforms.RandomCrop(448),
                       #transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        img_data = VOCDataset(root = file_root,list_file=listano,train=True,transform=transform_train,loader = cv_loader)
        train_loader = torch.utils.data.DataLoader(img_data, batch_size=batchsize,shuffle=True)
        self.train_loader = train_loader
        #self.img_data=img_data

        transform_test = transforms.Compose([
                        #transforms.Resize(448),
                        #transforms.CenterCrop(448),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        img_data_t = VOCDataset(root = file_root,list_file='voc2012.txt',train=False,transform=transform_test,loader = cv_loader)
        test_loader = torch.utils.data.DataLoader(img_data_t, batch_size=int(0.5*batchsize),shuffle=False)
        self.test_loader = test_loader
    
    def test(self):
        #print(len(self.img_data))
        print('there are total %s batches in training and total %s batches for test' % (len(self.train_loader),len(self.test_loader)))
        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            print( batch_x.size(), batch_y.size())
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            print( batch_x.size(), batch_y.size())
    
    def getdata(self):
        return self.train_loader, self.test_loader


if __name__ == '__main__':
    #testdata = Yolodata(file_root = '/home/claude.duan/data/VOCdevkit/VOC2012/JPEGImages/', listano = './voc2012.txt',batchsize=2)
    testdata = Yolodata(file_root = '/Users/duanyiqun/Downloads/VOCdevkit/VOC2012/JPEGImages/', listano = './voc2012.txt',batchsize=2)
    testdata.test()


