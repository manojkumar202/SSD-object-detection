import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    
    #ann_confidence = ann_confidence.long()
    
    #temp_ann_confidence = torch.argmax(temp_ann_confidence, axis=1)
    ann_confidence = ann_confidence.reshape(ann_confidence.shape[0]*ann_confidence.shape[1], ann_confidence.shape[2])
    pred_confidence = pred_confidence.reshape(pred_confidence.shape[0]*pred_confidence.shape[1], pred_confidence.shape[2])
    ann_box = ann_box.reshape(ann_box.shape[0]*ann_box.shape[1], ann_box.shape[2])
    pred_box = pred_box.reshape(pred_box.shape[0]*pred_box.shape[1], pred_box.shape[2])
    
    temp_ann_confidence = ann_confidence
    obj_ind = torch.where(ann_confidence[:,-1]==0)
    noobj_ind = torch.where(ann_confidence[:,-1]==1)
    
    obj_pred_confidence = pred_confidence[obj_ind] #torch.index_select(pred_confidence, 0, obj_ind).cpu()
    obj_ann_confidence = temp_ann_confidence[obj_ind] #torch.index_select(temp_ann_confidence, 0, obj_ind).cpu()
    obj_ann_confidence = torch.argmax(obj_ann_confidence, axis=1)
    #print(obj_ann_confidence)
    
    noobj_pred_confidence = pred_confidence[noobj_ind] #torch.index_select(pred_confidence, 0, noobj_ind).cpu()
    noobj_ann_confidence = temp_ann_confidence[noobj_ind] #torch.index_select(temp_ann_confidence, 0, noobj_ind).cpu()
    noobj_ann_confidence = torch.argmax(noobj_ann_confidence, axis=1)
    #print(noobj_ann_confidence)
    obj_pred_box = pred_box[obj_ind] #torch.index_select(pred_box, 0, obj_ind).cpu()
    obj_ann_box = ann_box[obj_ind] #torch.index_select(ann_box, 0, obj_ind).cpu()
    
    
    #print(obj_pred_confidence.shape)
    #print(obj_ann_confidence.shape)
    conf_loss = F.cross_entropy(obj_pred_confidence, obj_ann_confidence) + 3* F.cross_entropy(noobj_pred_confidence, noobj_ann_confidence)
    
    #obj_pred_box = pred_box[obj_ind]
    #obj_ann_box = ann_box[obj_ind]
    box_loss = F.smooth_l1_loss(obj_pred_box, obj_ann_box)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    return conf_loss +3.5* box_loss



class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.conv1 = nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU() )
        self.conv2 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU() )
        self.conv3 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU() )
        self.conv4 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU() )
        self.conv5 = nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU() )
        self.conv6 = nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU() )
        self.conv7 = nn.Sequential( nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv8 = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv9 = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv10 = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU() )
        self.conv11 = nn.Sequential( nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU() )
        self.conv12 = nn.Sequential( nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU() )
        self.conv13 = nn.Sequential( nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU() )
        #first split
        self.conv_sub1_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv_sub1_rleft = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_sub1_rright = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        #down layer
        self.conv_straight1_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU() )
        #second split
        self.conv_sub2_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv_sub2_rleft = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_sub2_rright = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        #down layer
        self.conv_straight2_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU() )
        #third split
        self.conv_sub3_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU() )
        self.conv_sub3_rleft = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_sub3_rright = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        #down layer
        self.conv_straight3_left = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1), nn.BatchNorm2d(256), nn.ReLU() )
        
        self.conv_final_left = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)
        self.conv_final_right = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1)
        #self.smax = nn.Softmax(dim=1)
        #TODO: define layers
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)
        x = self.conv6(x)
        #print(x.shape)
        x = self.conv7(x)
        #print(x.shape)
        x = self.conv8(x)
        #print(x.shape)
        x = self.conv9(x)
        #print(x.shape)
        x = self.conv10(x)
        #print(x.shape)
        x = self.conv11(x)
        #print(x.shape)
        x = self.conv12(x)
        #print(x.shape)
        x = self.conv13(x)
        #print(x.shape)

        first_lsplit_x = self.conv_sub1_left(x)
        #print("first_lsplit_x  ", first_lsplit_x.shape)
        first_straight_x = self.conv_straight1_left(first_lsplit_x)
        #print("first_straight_x  ", first_straight_x.shape)

        second_lsplit_x = self.conv_sub2_left(first_straight_x)
        #print("second_lsplit_x  ", second_lsplit_x.shape)
        second_straight_x = self.conv_straight2_left(second_lsplit_x)
        #print("second_straight_x  ", second_straight_x.shape)

        third_lsplit_x = self.conv_sub3_left(second_straight_x)
        #print("third_lsplit_x  ", third_lsplit_x.shape)
        third_straight_x = self.conv_straight3_left(third_lsplit_x)
        #print("third_straight_x  ", third_straight_x.shape)

        final_left_x = self.conv_final_left(third_straight_x)
        #print("final_left_x  ", final_left_x.shape)
        final_right_x = self.conv_final_right(third_straight_x)
        #print("final_right_x  ", final_right_x.shape)
        
        first_rsplit_left_x = self.conv_sub1_rleft(x)
        #print("first_rsplit_left_x  ", first_rsplit_left_x.shape)
        first_rsplit_right_x = self.conv_sub1_rleft(x)
        #print("first_rsplit_right_x  ", first_rsplit_right_x.shape)

        second_rsplit_left_x = self.conv_sub2_rleft(first_straight_x)
        #print("second_rsplit_left_x  ", second_rsplit_left_x.shape)
        second_rsplit_right_x = self.conv_sub2_rleft(first_straight_x)
        #print("second_rsplit_right_x  ", second_rsplit_right_x.shape)

        third_rsplit_left_x = self.conv_sub3_rleft(second_straight_x)
        #print("third_rsplit_left_x  ", third_rsplit_left_x.shape)
        third_rsplit_right_x = self.conv_sub3_rleft(second_straight_x)
        #print("third_rsplit_right_x  ", third_rsplit_right_x.shape)

        layer1_rleft_reshape = first_rsplit_left_x.reshape( first_rsplit_left_x.shape[0], 16, 100)
        #print("layer1_rleft_reshape  ", layer1_rleft_reshape.shape)
        layer1_rright_reshape = first_rsplit_right_x.reshape( first_rsplit_right_x.shape[0], 16, 100)
        #print("layer1_rright_reshape  ", layer1_rright_reshape.shape)
        
        layer2_rleft_reshape = second_rsplit_left_x.reshape( second_rsplit_left_x.shape[0], 16, 25)
        #print("layer2_rleft_reshape  ", layer2_rleft_reshape.shape)
        layer2_rright_reshape = second_rsplit_right_x.reshape( second_rsplit_right_x.shape[0], 16, 25)
        #print("layer2_rright_reshape  ", layer2_rright_reshape.shape)
        
        layer3_rleft_reshape = third_rsplit_left_x.reshape( third_rsplit_left_x.shape[0], 16, 9)
        #print("layer3_rleft_reshape  ", layer3_rleft_reshape.shape)
        layer3_rright_reshape = third_rsplit_right_x.reshape( third_rsplit_right_x.shape[0], 16, 9)
        #print("layer3_rright_reshape  ", layer3_rright_reshape.shape)
        
        layer3_finalleft_reshape = final_left_x.reshape( final_left_x.shape[0], 16, 1)
        #print("layer3_finalleft_reshape  ", layer3_finalleft_reshape.shape)
        layer3_finalright_reshape = final_right_x.reshape( final_right_x.shape[0], 16, 1)
        #print("layer3_finalright_reshape  ", layer3_finalright_reshape.shape)

        bboxes = torch.cat((layer1_rleft_reshape, layer2_rleft_reshape, layer3_rleft_reshape, layer3_finalleft_reshape),2)
        #print("bboxes  ", bboxes.shape)
        bboxes = bboxes.permute(0,2,1)
        #print("bboxes  ", bboxes.shape)
        bboxes = bboxes.reshape(bboxes.shape[0], 540, self.class_num)
        #print("bboxes  ", bboxes.shape)

        confidence = torch.cat((layer1_rright_reshape, layer2_rright_reshape, layer3_rright_reshape, layer3_finalright_reshape),2)
        #print("confidence  ", confidence.shape)
        confidence = confidence.permute(0,2,1)
        #print("confidence  ", confidence.shape)
        confidence = confidence.reshape(confidence.shape[0], 540, self.class_num)
        #print("confidence  ", confidence.shape)
        #confidence = self.smax(confidence)
        #print("confidence  ", confidence.shape)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










