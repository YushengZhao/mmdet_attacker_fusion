# coding=utf-8
import cv2
import torch
import numpy as np

def six2fiv(flag):
    NewR = 28
    flag_m = 21
    flag_n = 21
    mask = np.zeros([500,500,3])
    for i in range(flag_m):
        for j in range(flag_n):
            # 计算每个块的起始位置
            if (flag[(i * flag_n + j)] == 1.0):
                c_x = int((i * NewR+13)*0.82236)
                c_y = int((j * NewR+13)*0.82236)
                mask[(c_x-11):(c_x+11),(c_y-11):(c_y+11),:]=1.0
    return mask


def InitialMask(Mask):
    m,n,_ = np.shape(Mask)    # 得到 mask 的长宽
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    flag = np.zeros((flag_m*flag_n))   # 记录这些块的状态，初始时都是被选中的
    # 根据bboxes信息确定patch的大概位置（加速计算）
    for i in range(flag_m):
        for j in range(flag_n):
            # 计算每个块的起始位置
            start_x = i*DR
            start_y = j*DR
            Mask[start_x:(start_x+DR),start_y:(start_y+DR),:] = 1.0
            flag[(i*flag_n +j)] = 1.0
    return flag, Mask


def SampleMask(flag, Mask, NOISE):
    m,n,_ = np.shape(Mask)  # 得到 mask 的长宽
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    N = flag_n*flag_m  #块儿数
    idx = np.zeros(N)   # 用来记录每个块的重要程度，越大越不重要
    #根据像素绝对扰动量大小排序
    for i in range(N):
        if flag[i]<1.0:
            idx[i] = 0
        else:
            x = int(i/flag_n)*DR
            y = int(i%flag_n)*DR
            idx[i] = np.mean(np.abs(NOISE[x:(x+DR),y:(y+DR),:]))
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    f = np.zeros((flag_m * flag_n))
    mask = np.zeros_like(Mask)
    half = int(0.5 * np.sum(flag))
    if half <= 10:
        for j in range(10):
            f[idx_sort[j]] = 1.0
            x = int(idx_sort[j] / flag_n) * DR
            y = int(idx_sort[j] % flag_n) * DR
            mask[x:(x + DR), y:(y + DR), :] = 1.0
    else:
        for j in range(half):
            f[idx_sort[j]] = 1.0
            x = int(idx_sort[j] / flag_n) * DR
            y = int(idx_sort[j] % flag_n) * DR
            mask[x:(x + DR), y:(y + DR),:] = 1.0
    return f, mask

def FinalMask(flag, Mask, NOISE):
    m,n,_ = np.shape(Mask)  # 得到 mask 的长宽
    P_pixels = m*n*0.02       # 总共可以扰动的像素数
    radius = int(0.5*np.sqrt(P_pixels/10))  # 最大块的半径
    # 计算 mask 可以切多少个块（假定块中间空1个像素）
    DR = 2*radius   # 块的宽（正方形）
    flag_m = int(m/DR)
    flag_n = int(n/DR)
    N = flag_n*flag_m  #块儿数
    idx = np.zeros(N)   # 用来记录每个块的重要程度，越大越不重要
    #根据像素绝对扰动量大小排序
    for i in range(N):
        if flag[i]<1.0:
            idx[i] = 0
        else:
            x = int(i/flag_n)*DR
            y = int(i%flag_n)*DR
            idx[i] = np.mean(np.abs(NOISE[x:(x+DR),y:(y+DR),:]))
    idx_sort = np.argsort(-idx)   # 返回从小到大的下坐标
    # 重新 设置flag以及mask
    f = np.zeros((flag_m * flag_n))
    mask = np.zeros_like(Mask)
    for j in range(10):
        f[idx_sort[j]] = 1.0
        x = int(idx_sort[j] / flag_n) * DR
        y = int(idx_sort[j] % flag_n) * DR
        mask[x:(x + DR), y:(y + DR),:] = 1.0
    return f, mask