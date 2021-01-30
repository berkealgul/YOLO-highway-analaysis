from __future__ import division

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import random


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = T.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  T.max(b1_x1, b2_x1)
    inter_rect_y1 =  T.max(b1_y1, b2_y1)
    inter_rect_x2 =  T.min(b1_x2, b2_x2)
    inter_rect_y2 =  T.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = T.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * T.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def predict_transform(prediction, im_dims, anchors, class_c, CUDA=True):
    batch_size = prediction.size(0)
    stride = im_dims // prediction.size(2)
    grid_size = im_dims // stride
    anchor_c = len(anchors)
    box_atts = 5 + class_c

    # Reshaping
    prediction = prediction.view(batch_size, anchor_c*box_atts, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*anchor_c, box_atts)

    # Anchor adjustment
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Normalize c_x c_y and object score
    prediction[:,:,0] = T.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = T.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = T.sigmoid(prediction[:,:,4])

    # Offser x and y
    grid = np.arange(grid_size)
    i, j = np.meshgrid(grid, grid)

    x = T.FloatTensor(i).view(-1, 1)
    y = T.FloatTensor(j).view(-1, 1)

    if CUDA:
        x = x.cuda()
        y = y.cuda()

    offset = T.cat((x, y), 1).repeat(1, anchor_c).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += offset

    #Apply anchor dims
    anchors = T.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = T.exp(prediction[:,:,2:4])*anchors

    #Normalize class scores
    prediction[:,:,5: 5+class_c] = T.sigmoid((prediction[:,:, 5:5+class_c]))

    prediction[:,:,:4] *= stride

    return prediction


def adjust_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = T.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = T.cat(seq, 1)

        non_zero_ind =  (T.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index

        for cls in img_classes:
            #perform NMS
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = T.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = T.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries
                non_zero_ind = T.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = T.cat(seq,1)
                write = True
            else:
                out = T.cat(seq,1)
                output = T.cat((output,out))

    try:
        return output
    except:
        return 0


def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image

    return canvas


def prep_image(img, inp_dim):
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = T.from_numpy(img).float().div(255.0).unsqueeze(0)
    img = Variable(img)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def create_colors(num_classes):
    colors = [(random.randrange(0, 125), random.randrange(0, 125),
                random.randrange(0, 125)) for i in range(num_classes)]
    return colors


def write_result(result, img, in_dim, classes, colors, valid_classes=None):
    coordinates = list()

    scale = in_dim / max(img.shape)
    offsetX = int((in_dim-img.shape[1]*scale)/2)
    offsetY = int((in_dim-img.shape[0]*scale)/2)
    result[:,[1,3]] = (result[:,[1,3]] - offsetX)/scale
    result[:,[2,4]] = (result[:,[2,4]] - offsetY)/scale

    for i in range(result.shape[0]):
        result[i, [1,3]] = T.clamp(result[i, [1,3]], 0.0, img.shape[1])
        result[i, [2,4]] = T.clamp(result[i, [2,4]], 0.0, img.shape[0])

    for cell in result:
        c1 = tuple((cell[1:3]).int())
        c2 = tuple((cell[3:5]).int())

        cls = int(cell[-1])
        color = colors[cls]

        label = "{0}".format(classes[cls])

        if cls not in valid_classes:
            return

        """
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color,-1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        """
        coordinates.append((c1, c2))

    return coordinates
