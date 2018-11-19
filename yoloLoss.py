import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import torchvision.models as models


class YoloLoss(nn.Module):
    def __init__(self, n_batch, B, C, lambda_coord, lambda_noobj, use_gpu=False):
        """
        :param n_batch: number of batches
        :param B: number of bounding boxes
        :param C: number of bounding classes
        :param lambda_coord: factor for loss which contain objects
        :param lambda_noobj: factor for loss which do not contain objects
        """
        super(YoloLoss, self).__init__()
        self.n_batch = n_batch
        self.B = B # assume there are two bounding boxes
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.use_gpu = use_gpu

    def compute_iou(self, bbox1, bbox2):
        """
        Compute the intersection over union of two set of boxes, each box is [x_center,y_center,w,h]
        :param bbox1: (tensor) bounding boxes, size [N,4]
        :param bbox2: (tensor) bounding boxes, size [M,4]
        :return:
        """
        # transfer center cordinate to x1,y1,x2,y2
        b1x1y1 = bbox1[:,:2]-bbox1[:,2:]**2 # [N, (x1,y1)=2]
        b1x2y2 = bbox1[:,:2]+bbox1[:,2:]**2 # [N, (x2,y2)=2]
        b2x1y1 = bbox2[:,:2]-bbox2[:,2:]**2 # [M, (x1,y1)=2]
        b2x2y2 = bbox2[:,:2]+bbox2[:,2:]**2 # [M, (x1,y1)=2]
        box1 = torch.cat((b1x1y1.view(-1,2), b1x2y2.view(-1, 2)), dim=1) # [N,4], 4=[x1,y1,x2,y2]
        box2 = torch.cat((b2x1y1.view(-1,2), b2x2y2.view(-1, 2)), dim=1) # [M,4], 4=[x1,y1,x2,y2]
        N = box1.size(0)
        M = box2.size(0)
        # find cordinate of intersaction boxes.
        tl = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        br = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        
        #   width and height
        wh = br - tl  # [N,M,2]
        wh[(wh<0).detach()] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        :param pred_tensor: [batch,SxSx(Bx5+20))]
        :param target_tensor: [batch,S,S,Bx5+20]
        :return: total loss
        """
        n_elements = self.B * 5 + self.C
        batch = target_tensor.size(0)
        target_tensor = target_tensor.view(batch,-1,n_elements)

        pred_tensor = pred_tensor.view(batch,-1,n_elements)
        coord_mask = target_tensor[:,:,5] > 0
        noobj_mask = target_tensor[:,:,5] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        coord_target = target_tensor[coord_mask].view(-1,n_elements)
        coord_pred = pred_tensor[coord_mask].view(-1,n_elements)
        # seperately take all these cordinates and 
        class_pred = coord_pred[:,self.B*5:]
        class_target = coord_target[:,self.B*5:]
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)

        noobj_target = target_tensor[noobj_mask].view(-1,n_elements)
        noobj_pred = pred_tensor[noobj_mask].view(-1,n_elements)

        # compute loss which do not contain objects
        if self.use_gpu:
            noobj_target_mask = torch.cuda.ByteTensor(noobj_target.size())
        else:
            noobj_target_mask = torch.ByteTensor(noobj_target.size())
        noobj_target_mask.zero_()
        for i in range(self.B):
            noobj_target_mask[:,i*5+4] = 1
        noobj_target_c = noobj_target[noobj_target_mask] # only compute loss of c size [2*B*noobj_target.size(0)]
        noobj_pred_c = noobj_pred[noobj_target_mask]
        noobj_loss = functional.mse_loss(noobj_pred_c, noobj_target_c, size_average=False)

        # compute loss which contain objects
        if self.use_gpu:
            coord_response_mask = torch.cuda.ByteTensor(box_target.size())
            coord_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        else:
            coord_response_mask = torch.ByteTensor(box_target.size())
            coord_not_response_mask = torch.ByteTensor(box_target.size())
        coord_response_mask.zero_()
        coord_not_response_mask = ~coord_not_response_mask.zero_()
        for i in range(0,box_target.size()[0],self.B):
            box1 = box_pred[i:i+self.B]
            box2 = box_target[i:i+self.B]
            iou = self.compute_iou(box1[:, :4], box2[:, :4])
            max_iou, max_index = iou.max(0)
            if self.use_gpu:
                max_index = max_index.data.cuda()
            else:
                max_index = max_index.data
            coord_response_mask[i+max_index]=1
            coord_not_response_mask[i+max_index]=0

        # 1. response loss
        box_pred_response = box_pred[coord_response_mask].view(-1, 5)
        box_target_response = box_target[coord_response_mask].view(-1, 5)
        contain_loss = functional.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
        loc_loss = functional.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) +\
                   functional.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4], size_average=False)
        # 2. not response loss
        box_pred_not_response = box_pred[coord_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coord_not_response_mask].view(-1, 5)

        # compute class prediction loss
        class_loss = functional.mse_loss(class_pred, class_target, size_average=False)

        # compute total loss
        total_loss = self.lambda_coord * loc_loss + contain_loss + self.lambda_noobj * noobj_loss + class_loss
        return total_loss
    


