# coding=utf-8
import os
import sys

sys.path.append('./mmdetection/')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from mmdet.apis import init_detector
import cv2
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from util_copy.utils import *
from tool.darknet2pytorch import *
from attack_utils.attackloss import ada_attack

# yolo
cfgfile = "models/yolov4.cfg"
weightfile = "models/yolov4.weights"
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

# faster rcnn
config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
rcnn_model = init_detector(config, checkpoint, device='cuda:1')  # 构建 faster rcnn

# 循环攻击目录中的每张图片
clean_path = 'select1000_new/'  # 干净图片目录
dirty_path = 'select1000_new_p/'  # 对抗图片存放位置
dirty_path2 = 'select1000_new_p2/'
imgs_list = os.listdir(clean_path)
imgs_list.sort()


def get_mesh(mask, threshold=2000):
    cnt = np.sum(mask) / 3
    stride = 1
    f = lambda x: (2 * x - 1) / (x * x)
    while cnt * f(stride) >= threshold:
        stride += 1
    mesh = np.zeros((500, 500, 3))
    mesh[stride//2::stride, :, :] = 1
    mesh[:, stride//2::stride, :] = 1
    return mesh


# imgs_list = imgs_list[500:]
for i in range(len(imgs_list)):
    image_name = os.path.basename(imgs_list[i]).split('.')[0]  # 测试图片名称
    print('It is attacking on the {}-th image, the image name is {}'.format(i, image_name))
    image_path = os.path.join(clean_path, imgs_list[i])
    img = cv2.imread(image_path)
    mask = np.load('Mask/mask2900/{}.npy'.format(image_name))
    # mesh = get_mesh(mask)
    # mask = mask * mesh
    # finalimg, noise = str_attack(darknet_model, img, conf_thresh=0.35, max_iter=120, epsilon=10, mask=mask)
    finalimgs, noise = ada_attack(darknet_model, rcnn_model, img, conf_thresh=0.35, max_iter=250, epsilon=6, mask=mask)
    # finalimg, noise = gen_attack(darknet_model, img, conf_thresh=0.35, max_iter=100, epsilon=2, mask=mask)
    image_pert_path = os.path.join(dirty_path, imgs_list[i])
    image_pert_path2 = os.path.join(dirty_path2, imgs_list[i])
    if len(finalimgs) == 1:
        finalimg = finalimgs[0]
        finalimg.save(image_pert_path)
        finalimg.save(image_pert_path2)
    else:
        finalimgs[-1].save(image_pert_path)
        finalimgs[-2].save(image_pert_path2)
    # cv2.imwrite(,finalimg)
print('done...')
