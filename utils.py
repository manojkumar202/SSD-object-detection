import numpy as np
import cv2
from dataset import iou
import math
import torch
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, ogn_pred_box, final_bound_box_class=[]):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_*255
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:].copy()
    image2[:]=image[:].copy()
    image3[:]=image[:].copy()
    image4[:]=image[:].copy()
    color_ind=[0,1,2]
    if windowname=="inference":
        #print("INSIDE INFERENCE")
        t_final_bound_box_class=[]
        pred_confidence = softmax(pred_confidence, axis=1)
        for i in range(len(pred_confidence)):
            for j in range(class_num):
                if pred_confidence[i,j]>0.25:
                    #TODO:
                    #image3: draw network-predicted bounding boxes on image3
                    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                    #print("pred box", pred_box[i])
                    #print("ann box", ann_box[i])
                    gx = boxs_default[i][2]*pred_box[i][0] + boxs_default[i][0]
                    gy = boxs_default[i][3]*pred_box[i][1] + boxs_default[i][1]
                    gw = boxs_default[i][2]*math.exp(pred_box[i][2])
                    gh = boxs_default[i][3]*math.exp(pred_box[i][3])
                    #print("gx, gy, gw, gh", gx, "  ", gy, "  ", gw, "  ", gh, "  ")
                    start_point = ( int( (gx-(gw/2)) *320), int( (gy-(gh/2)) *320) )
                    end_point = ( int( (gx+(gw/2)) *320), int( (gy+(gh/2)) *320)  )
                    color=[0,0,0]
                    color[color_ind[j]] = 255
                    thickness = 2
                    cv2.rectangle(image1, start_point, end_point, color, thickness)
                    t_final_bound_box_class.append([j, int( (gx-(gw/2)) *320), int( (gy-(gh/2)) *320), gw, gh])
                #if ogn_pred_box[i, j]>0.25:
                    #start_point = ( int( (pred_box[i][0]-(pred_box[i][2]/2))*320 ), int( (pred_box[i][1]-(pred_box[i][3]/2))*320 ) )
                    #end_point = ( int( (pred_box[i][0]+(pred_box[i][2]/2))*320), int( (pred_box[i][1]+(pred_box[i][3]/2))*320 ) )
                    start_point = ( int( ( boxs_default[i][0] - (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] - (boxs_default[i][3]/2) )*320 ) )
                    end_point = ( int( ( boxs_default[i][0] + (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] + (boxs_default[i][3]/2) )*320 ) )
                    color=[0,0,0]
                    color[color_ind[j]] = 255
                    thickness = 2
                    cv2.rectangle(image2, start_point, end_point, color, thickness)
        h,w,_ = image1.shape
        image = np.zeros([h,w*2,3], np.uint8)
        image[:,:w] = image1
        image[:,w:] = image2
        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
        #cv2.waitKey(1)
        plt.imshow(image)
        plt.show()
        #print("COMPLETED INFERENCE")
        return final_bound_box_class.append(t_final_bound_box_class)
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                gx = boxs_default[i][2]*ann_box[i][0] + boxs_default[i][0]
                #print("boxs_defaultx , y, w, h", boxs_default[i][0], "  ", boxs_default[i][1], "  ", boxs_default[i][2], "  ",boxs_default[i][3], "  ")
                gy = boxs_default[i][3]*ann_box[i][1] + boxs_default[i][1]
                gw = boxs_default[i][2]*math.exp(ann_box[i][2])
                gh = boxs_default[i][3]*math.exp(ann_box[i][3])
                #print("gx, gy, gw, gh", gx, "  ", gy, "  ", gw, "  ", gh, "  ")
                start_point = ( int( (gx-(gw/2)) *320), int( (gy-(gh/2)) *320) )
                end_point = ( int( (gx+(gw/2)) *320), int( (gy+(gh/2)) *320)  )
                color=[0,0,0]
                color[color_ind[j]] = 255
                thickness = 2
                cv2.rectangle(image1, start_point, end_point, color, thickness)

                start_point = ( int( ( boxs_default[i][0] - (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] - (boxs_default[i][3]/2) )*320 ) )
                end_point = ( int( ( boxs_default[i][0] + (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] + (boxs_default[i][3]/2) )*320 ) )
                color=[0,0,0]
                color[color_ind[j]] = 255
                thickness = 2
                cv2.rectangle(image2, start_point, end_point, color, thickness)
    
    #pred
    pred_confidence = softmax(pred_confidence, axis=1)
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.37:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                #print("pred box", pred_box[i])
                #print("ann box", ann_box[i])
                gx = boxs_default[i][2]*pred_box[i][0] + boxs_default[i][0]
                gy = boxs_default[i][3]*pred_box[i][1] + boxs_default[i][1]
                gw = boxs_default[i][2]*math.exp(pred_box[i][2])
                gh = boxs_default[i][3]*math.exp(pred_box[i][3])
                #print("gx, gy, gw, gh", gx, "  ", gy, "  ", gw, "  ", gh, "  ")
                start_point = ( int( (gx-(gw/2)) *320), int( (gy-(gh/2)) *320) )
                end_point = ( int( (gx+(gw/2)) *320), int( (gy+(gh/2)) *320)  )
                color=[0,0,0]
                color[color_ind[j]] = 255
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)
                
            #if ogn_pred_box[i, j]>0.3:
                #start_point = ( int( (pred_box[i][0]-(pred_box[i][2]/2))*320 ), int( (pred_box[i][1]-(pred_box[i][3]/2))*320 ) )
                #end_point = ( int( (pred_box[i][0]+(pred_box[i][2]/2))*320), int( (pred_box[i][1]+(pred_box[i][3]/2))*320 ) )
                start_point = ( int( ( boxs_default[i][0] - (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] - (boxs_default[i][3]/2) )*320 ) )
                end_point = ( int( ( boxs_default[i][0] + (boxs_default[i][2]/2) )*320 ), int( ( boxs_default[i][1] + (boxs_default[i][3]/2) )*320 ) )
                color=[0,0,0]
                color[color_ind[j]] = 255
                thickness = 2
                cv2.rectangle(image4, start_point, end_point, color, thickness)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    #cv2.waitKey(1)
    plt.imshow(image)
    plt.show()
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.3):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    maxprobpossible = 1    
    iou_pred_box = np.zeros([540,8], np.float32)
    
    for i in range(540):
        iou_pred_box[i][0] = boxs_default[i][2]*box_[i][0] + boxs_default[i][0]
        iou_pred_box[i][1] = boxs_default[i][3]*box_[i][1] + boxs_default[i][1]
        iou_pred_box[i][2] = boxs_default[i][2]*math.exp(box_[i][2])
        iou_pred_box[i][3] = boxs_default[i][3]*math.exp(box_[i][3])
        iou_pred_box[i][4] = iou_pred_box[i][0]-(iou_pred_box[i][2]/2)
        iou_pred_box[i][5] = iou_pred_box[i][1]-(iou_pred_box[i][3]/2)
        iou_pred_box[i][6] = iou_pred_box[i][0]+(iou_pred_box[i][2]/2)
        iou_pred_box[i][7] = iou_pred_box[i][1]+(iou_pred_box[i][3]/2)
    ogn_pred_box = box_.copy()
    final_box = box_.copy() #np.zeros([box_.shape[0],4], np.float32)
    final_conf = confidence_.copy() #np.zeros([confidence_.shape[0],4], np.float32)
    confidence_ = softmax(confidence_, axis=1)
    B = []
    while maxprobpossible:
        
        check = 0
        max_value = -1
        max_index = -1
        max_inds_class = -1
        for i in range(len(box_)):
            if i not in B:
                #print(final_conf[i][:3])
                t_max_index = i
                t_max_value = np.max(final_conf[i][:3])
                t_max_inds_class = np.argmax(final_conf[i][:3])
                if t_max_value>max_value and t_max_value>threshold:
                    max_value = t_max_value
                    max_index = t_max_index
                    max_inds_class = t_max_inds_class
                    check=1
        #print(max_value)
        #print(max_index)
        if check==0:
            maxprobpossible=0
        else:
            B.append(max_index)
            #print(max_index)
            ious = iou(iou_pred_box, iou_pred_box[max_index][4], iou_pred_box[max_index][5], iou_pred_box[max_index][6], iou_pred_box[max_index][7] )
            #print(ious_true[max_index])
            ious_true = ious>overlap
            for i in range(len(ious_true)):
                t_max_inds_class = np.argmax(final_conf[i][:3])
                if i not in B and ious_true[i] and max_inds_class==t_max_inds_class:
                    #print()
                    final_conf[i][0] = 0
                    final_conf[i][1] = 0
                    final_conf[i][2] = 0
                    final_conf[i][3] = 1
                    final_box[i] = [0, 0, 0, 0]
        #print(B)
        #print(maxprobpossible)
    return final_conf, final_box, ogn_pred_box

def generate_mAP(pred_confidence, pred_box, boxs_default, val_gt_info, overlap=0.5, threshold=0.3):
    #TODO: Generate mAP
    num_classes=3
    #val_gt_info = val_gt_info.reshape(val_gt_info.shape[0]*val_gt_info.shape[1], val_gt_info.shape[2])
    #ann_box = ann_box.reshape(ann_box.shape[0]*ann_box.shape[1], ann_box.shape[2])
    #pred_box = pred_box.reshape(pred_box.shape[0]*pred_box.shape[1], pred_box.shape[2])
    #ann_confidence = ann_confidence.reshape(ann_confidence.shape[0]*ann_confidence.shape[1], ann_confidence.shape[2])
    #pred_confidence = pred_confidence.reshape(pred_confidence.shape[0]*pred_confidence.shape[1], pred_confidence.shape[2])
    #val_gt_info = val_gt_info.reshape(val_gt_info.shape[0]*val_gt_info.shape[1], val_gt_info.shape[2])
    dataframes_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    #score_list = []
    #print(val_gt_info)
    pred_confidence = softmax(pred_confidence, axis=1)
    for i in range(len(pred_box)):
        pred_box[i][0] = boxs_default[i][2]*pred_box[i][0] + boxs_default[i][0]
        pred_box[i][1] = boxs_default[i][3]*pred_box[i][1] + boxs_default[i][1]
        pred_box[i][2] = boxs_default[i][2]*math.exp(pred_box[i][2])
        pred_box[i][3] = boxs_default[i][3]*math.exp(pred_box[i][3])
    #print(pred_confidence[0])
    for i in range(num_classes):
        obj_ind = np.where(pred_confidence[:,i]!=0)
        #print(len(obj_ind[0]))
        obj_true_positive = np.zeros(len(obj_ind[0]), np.float32)
        obj_false_positive = np.zeros(len(obj_ind[0]), np.float32)
        obj_pred_box = pred_box[obj_ind]
        
        obj_pred_confidence = pred_confidence[obj_ind][:,i]
        #obj_val_gt_info = val_gt_info
        for j in range(len(obj_pred_box)):
            one_t_one_check = 0 
            i_cls_count = 0
            for line in val_gt_info:
                class_id = line[0]
                org_height = line[5]
                org_width = line[6]
                gx = line[1]/ org_width
                gy = line[2]/ org_height
                gw = line[3]/ org_width
                gh = line[4]/ org_height
                x_min = gx 
                y_min = gy 
                x_max = gx+gw 
                y_max = gy+gh

                
                if obj_pred_confidence[j]>0.5:
                    #print("GT X_MIN ", class_id, x_min, y_min, x_max, y_max)
                    pred_box_single = np.array([[obj_pred_box[j][0], obj_pred_box[j][1], obj_pred_box[j][2], obj_pred_box[j][3], 
                      obj_pred_box[j][0]-(obj_pred_box[j][2]/2), obj_pred_box[j][1]-(obj_pred_box[j][3]/2), 
                      obj_pred_box[j][0]+(obj_pred_box[j][2]/2), obj_pred_box[j][1]+(obj_pred_box[j][3]/2)]]) 
                    #print("PT X_MIN ",class_id, obj_pred_box[j][0]-(obj_pred_box[j][2]/2), obj_pred_box[j][1]-(obj_pred_box[j][3]/2), 
                      #obj_pred_box[j][0]+(obj_pred_box[j][2]/2), obj_pred_box[j][1]+(obj_pred_box[j][3]/2))
                else:
                    pred_box_single = np.array([[obj_pred_box[i][0], obj_pred_box[i][1], obj_pred_box[i][2], obj_pred_box[i][3], 
                      obj_pred_box[j][0]-(obj_pred_box[j][2]/2), obj_pred_box[j][1]-(obj_pred_box[j][3]/2), 
                      obj_pred_box[j][0]+(obj_pred_box[j][2]/2), obj_pred_box[j][1]+(obj_pred_box[j][3]/2)]]) 
                    
                ious = iou(pred_box_single, x_min,y_min,x_max,y_max)
                if ious>overlap:
                    #print("came")
                    one_t_one_check = one_t_one_check+1
                    if i==class_id:
                        i_cls_count = i_cls_count + 1
            
            if one_t_one_check==1 and len(val_gt_info)==1 and i_cls_count==1:
                #print(one_t_one_check, len(val_gt_info), i_cls_count)
                obj_true_positive[j] = 1
            else:
                obj_false_positive[j] = 1
        #print(obj_pred_confidence.shape, obj_true_positive.shape, obj_false_positive.shape)
        
        t_df = np.column_stack((obj_pred_confidence, obj_true_positive, obj_false_positive))
                
        #t_df = np.concatenate((obj_pred_box, obj_pred_confidence), axis=1)
        dataframes_list[i]= pd.concat((dataframes_list[i], pd.DataFrame(t_df)))
    return dataframes_list








