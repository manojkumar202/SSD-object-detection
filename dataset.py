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
import torchvision.transforms.functional as TF
import random
import numpy as np
import os
import cv2, math
from PIL import Image

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:     
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = []
    for i in range(len(layers)):
        for cell in range(layers[i]*layers[i]):
            
            x_center = ( (cell%layers[i])+0.5 )/layers[i] 
            y_center = ( (cell//layers[i])+0.5 )/layers[i]
            
            lsize = large_scale[i]
            ssize = small_scale[i]
            width_bb = ssize
            height_bb = ssize
            x_min = x_center-(width_bb/2)
            x_max = x_center+(width_bb/2)
            y_min =  y_center-(height_bb/2)
            y_max = y_center+(height_bb/2)
            if x_min<0:
                x_min = 0
            if x_max>0.9:
                x_max = 0.9
            if y_min<0:
                y_min = 0
            if y_max>0.9:
                y_max = 0.9
            boxes.append( [x_center, y_center, width_bb, height_bb, x_min, y_min, x_max, y_max] )
            
            width_bb = lsize
            height_bb = lsize
            x_min = x_center-(width_bb/2)
            x_max = x_center+(width_bb/2)
            y_min =  y_center-(height_bb/2)
            y_max = y_center+(height_bb/2)
            if x_min<0:
                x_min = 0
            if x_max>0.9:
                x_max = 0.9
            if y_min<0:
                y_min = 0
            if y_max>0.9:
                y_max = 0.9
            boxes.append( [x_center, y_center, width_bb, height_bb, x_min, y_min, x_max, y_max] )
            
            width_bb = lsize*(2**0.5)
            height_bb = lsize/(2**0.5)
            x_min = x_center-(width_bb/2)
            x_max = x_center+(width_bb/2)
            y_min =  y_center-(height_bb/2)
            y_max = y_center+(height_bb/2)
            if x_min<0:
                x_min = 0
            if x_max>0.9:
                x_max = 0.9
            if y_min<0:
                y_min = 0
            if y_max>0.9:
                y_max = 0.9
            boxes.append( [x_center, y_center, width_bb, height_bb, x_min, y_min, x_max, y_max] )
            
            width_bb = lsize/(2**0.5)
            height_bb = lsize*(2**0.5)
            x_min = x_center-(width_bb/2)
            x_max = x_center+(width_bb/2)
            y_min =  y_center-(height_bb/2)
            y_max = y_center+(height_bb/2)
            if x_min<0:
                x_min = 0
            if x_max>0.9:
                x_max = 0.9
            if y_min<0:
                y_min = 0
            if y_max>0.9:
                y_max = 0.9
            boxes.append( [x_center, y_center, width_bb, height_bb, x_min, y_min, x_max, y_max] )
            
    boxes = np.array(boxes)    

    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,class_id,x_min,y_min,x_max,y_max, line, org_width, org_height):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    gx = line[1]/org_width
    gy = line[2]/org_height
    gw = line[3]/org_width
    gh = line[4]/org_height
    gx = gx+(gw/2)
    gy = gy+(gh/2)
    
    #gx = gx+(gw/2)
    #gy = gy+(gh/2)
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    assign_label = 0 
    ious_true = ious>threshold
    #TODO:
    for i in range(len(ious_true)):
        if ious_true[i]:
            assign_label=1
            if class_id==0:
                ann_confidence[i][1] = 0
                ann_confidence[i][2] = 0
                ann_confidence[i][3] = 0
                ann_confidence[i][0] = 1
            elif class_id==1:
                ann_confidence[i][0] = 0
                ann_confidence[i][2] = 0
                ann_confidence[i][3] = 0
                ann_confidence[i][1] = 1
            elif class_id==2:
                ann_confidence[i][0] = 0
                ann_confidence[i][1] = 0
                ann_confidence[i][3] = 0
                ann_confidence[i][2] = 1
            
            box = boxs_default[i]
            px = box[0]
            py = box[1]
            pw = box[2]
            ph = box[3]
            ann_box[i][0] =  ( gx-px ) / pw #tx
            ann_box[i][1] =  ( gy-py ) / ph #ty
            ann_box[i][2] =  math.log(gw/pw) #tw
            ann_box[i][3] =  math.log(gh/ph) #th
            
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if assign_label==0:
        marea_ind = np.argmax(ious)
        max_position =np.argmax(ann_confidence[marea_ind])
        ann_confidence[marea_ind][3] = 0
        ann_confidence[marea_ind][int(class_id)] = 1
        box = boxs_default[marea_ind]
        px = box[0]
        py = box[1]
        pw = box[2]
        ph = box[3]
        ann_box[marea_ind][0] =  ( gx-px ) / pw #tx
        ann_box[marea_ind][1] =  ( gy-py ) / ph #ty
        ann_box[marea_ind][2] =  math.log(gw/pw) #tw
        ann_box[marea_ind][3] =  math.log(gh/ph) #th
    
    return ann_box, ann_confidence


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = "True", image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        self.img_tensor = []
        self.org_width_list = []
        self.org_height_list = []
        self.cat_count = 0
        self.dog_count = 0 
        self.person_count =0
        if train=="True": #training
            print("Training")
            self.img_names = self.img_names[:int(len(self.img_names)*0.9)]
            for index in range(len(self.img_names)):
                img_name = self.imgdir+self.img_names[index]
                #ann_name = self.anndir+self.img_names[index][:-3]+"txt"
                image = transforms.ToTensor()(Image.open(img_name))
                org_height = image.shape[1]
                self.org_height_list.append(org_height)
                org_width = image.shape[2]
                self.org_width_list.append(org_width)
                resize_t = transforms.Resize((self.image_size,self.image_size))
                image = resize_t(image)
                if image.shape[0]==1:
                    image = torch.cat([image, image, image], dim=0)
                self.img_tensor.append(image)
            self.ann_text_list = []
            self.ann_text_processed = []
            for index in range(len(self.img_names)):
                ann_text = open(self.anndir+self.img_names[index][:-3]+"txt").readlines()
                
                temp_ann_processed = []
                for line in ann_text:
                    temp_ann_processed.append( [float(i) for i in line.rstrip().split(' ')] )    
                    
                self.ann_text_processed.append(temp_ann_processed)

        elif train==False: #validation
            print("Validation")
            self.img_names = self.img_names[:int(len(self.img_names)*0.1)]
        
            for index in range(len(self.img_names)):
                img_name = self.imgdir+self.img_names[index]
                #ann_name = self.anndir+self.img_names[index][:-3]+"txt"
                image = transforms.ToTensor()(Image.open(img_name))
                org_height = image.shape[1]
                self.org_height_list.append(org_height)
                org_width = image.shape[2]
                self.org_width_list.append(org_width)
                resize_t = transforms.Resize((self.image_size,self.image_size))
                image = resize_t(image)
                if image.shape[0]==1:
                    image = torch.cat([image, image, image], dim=0)
                self.img_tensor.append(image)
            self.ann_text_list = []
            self.ann_text_processed = []
            for index in range(len(self.img_names)):
                ann_text = open(self.anndir+self.img_names[index][:-3]+"txt").readlines()
                
                temp_ann_processed = []
                for line in ann_text:
                    temp_processed = [float(i) for i in line.rstrip().split(' ')]
                    temp_processed.append(self.org_height_list[index])
                    temp_processed.append(self.org_width_list[index])
                    temp_ann_processed.append( temp_processed )    
                    class_id = temp_processed[0]
                    if class_id==0:
                        self.cat_count = self.cat_count + 1
                    elif class_id==1:
                        self.dog_count = self.dog_count + 1
                    elif class_id ==2:
                        self.person_count = self.person_count + 1
                    
                self.ann_text_processed.append(temp_ann_processed)
        else:
            print("inference")
            self.img_names = self.img_names[:int(len(self.img_names))]
        
            for index in range(len(self.img_names)):
                img_name = self.imgdir+self.img_names[index]
                #ann_name = self.anndir+self.img_names[index][:-3]+"txt"
                image = transforms.ToTensor()(Image.open(img_name))
                org_height = image.shape[1]
                self.org_height_list.append(org_height)
                org_width = image.shape[2]
                self.org_width_list.append(org_width)
                resize_t = transforms.Resize((self.image_size,self.image_size))
                image = resize_t(image)
                if image.shape[0]==1:
                    image = torch.cat([image, image, image], dim=0)
                self.img_tensor.append(image)
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def get_count(self):
        return [self.cat_count, self.dog_count, self.person_count]

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        #img_name = self.imgdir+self.img_names[index]
        #ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        #image = transforms.ToTensor()(Image.open(img_name))
        #org_height = image.shape[1]
        #org_width = image.shape[2]
        #resize_t = transforms.Resize((self.image_size,self.image_size))
        #image = resize_t(image)
        #if image.shape[0]==1:
        #    image = torch.cat([image, image, image], dim=0)
        #img_channel, img_width, img_height = image.shape
        org_height = self.org_height_list[index]
        org_width = self.org_width_list[index]
        image = self.img_tensor[index]
        if self.train=="test":
            return image, self.img_names[index]
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        ann_text_processed = self.ann_text_processed[index]
        #assigned_bb = [0 for i in range(self.box_num)]
        #ann_text = open(ann_name).readlines()
        #for line in ann_text:
        #    ann_text_processed.append( [float(i) for i in line.rstrip().split(' ')] )
       
        for line in ann_text_processed:
            class_id = line[0]
            
            gx = line[1]/ org_width
            gy = line[2]/ org_height
            gw = line[3]/ org_width
            gh = line[4]/ org_height
            x_min = gx 
            y_min = gy 
            x_max = gx+gw 
            y_max = gy+gh 
            ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_max, y_max, line, org_width, org_height)   

        #match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_max, y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        if self.train=="True":
            random_check = random.randint(0, 80)
            if random_check ==50:
                image = TF.adjust_gamma(image, 0.7)
            if random_check==40:
                jitter_tranforms = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = jitter_tranforms(image)
            return image, ann_box, ann_confidence
        return image, ann_box, ann_confidence, ann_text_processed
