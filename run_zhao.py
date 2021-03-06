# coding=utf-8
import os
import sys
sys.path.append('./mmdetection/')

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
from torchvision import transforms
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


from util_copy.utils import *
from tool.darknet2pytorch import *
import matplotlib.pyplot as plt
from attack_utils.attackloss import L2_attack,ada_attack

# yolo
cfgfile = "models/yolov4.cfg"
weightfile = "models/yolov4.weights"
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

#faster rcnn
config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
rcnn_model = init_detector(config, checkpoint, device='cuda:0')      # 构建 faster rcnn

# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
dirty_path = 'select1000_new_zhao/'  # 对抗图片存放位置
imgs_list = os.listdir(clean_path)
for i in range(len(imgs_list)):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img = cv2.imread(image_path)
    mask_z = np.load('mask_zhao/{}.npy'.format(image_name))
    mask = np.zeros([500,500,3])
    mask[:,:,0]=mask_z
    mask[:,:,1]=mask_z
    mask[:,:,2]=mask_z
    #finalimg, noise = str_attack(darknet_model, img, conf_thresh=0.35, max_iter=120, epsilon=10, mask=mask)
    finalimg, noise = ada_attack(darknet_model,rcnn_model, img, conf_thresh=0.4, max_iter=120, epsilon=6, mask=mask)
    #finalimg, noise = gen_attack(darknet_model, img, conf_thresh=0.35, max_iter=100, epsilon=2, mask=mask)
    image_pert_path = os.path.join(dirty_path, imgs_list[i])
    finalimg.save(image_pert_path)
    #cv2.imwrite(,finalimg)
print('done...')