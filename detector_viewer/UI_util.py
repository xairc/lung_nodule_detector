import SimpleITK as sitk
import numpy as np
import torch
import math
import time
import sys
import cv2

from scipy.ndimage.interpolation import zoom
from torch.autograd import Variable
from training.layers import nms

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing, progressBar, order=2):
    print (len(imgs.shape))
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        progressBar.setValue(40)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def split_data(data, stride, split_comber):
    print (data.shape[1:])
    nz, nh, nw = data.shape[1:]
    pz = int(np.ceil(float(nz) / stride)) * stride
    ph = int(np.ceil(float(nh) / stride)) * stride
    pw = int(np.ceil(float(nw) / stride)) * stride
    data = np.pad(data, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant', constant_values=0)

    xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, data.shape[1] / stride),
                             np.linspace(-0.5, 0.5, data.shape[2] / stride),
                             np.linspace(-0.5, 0.5, data.shape[3] / stride), indexing='ij')
    coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

    data, nzhw = split_comber.split(data)
    coord2, nzhw2 = split_comber.split(coord,
                                            side_len=split_comber.side_len / stride,
                                            max_stride=split_comber.max_stride / stride,
                                            margin=split_comber.margin / stride)
    assert np.all(nzhw == nzhw2)
    data = (data.astype(np.float32) - 128) / 128

    return torch.from_numpy(data), torch.from_numpy(coord2), np.array(nzhw)

def convert_prob(pbb):

    for label in pbb:
        pos_ori = label[1:4]
        radious_ori = label[4]
        #pos_ori = pos_ori + extendbox[:, 0]

        label[1:4] = pos_ori
        label[4] = radious_ori
        label[0] = sigmoid(label[0])
    return pbb

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_nodule(net, data, coord, nzhw, lbb, n_per_run, split_comber, get_pbb, progressBar):

    net.eval()

    total_label = 0
    total_candi = 0

    splitlist = list(range(0, len(data) + 1, n_per_run))

    if splitlist[-1] != len(data):
        splitlist.append(len(data))
    outputlist = []

    for i in range(len(splitlist) - 1):
        inputdata = Variable(data[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
        inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()
        output = net(inputdata, inputcoord)
        outputlist.append(output.data.cpu().numpy())
        progressBar.setValue(10 + (80/len(splitlist) * (i+1)))
    output = np.concatenate(outputlist, 0)
    output = split_comber.combine(output, nzhw=nzhw)

    # fps 1.215909091, sens 0.933333333, thres 0.371853054
    thresh = 0.371853054
    pbb, mask = get_pbb(output, thresh, ismask=True)

    pbb = pbb[pbb[:, 0].argsort()[::-1]]
    pbb_cand_list = []
    # check overlap under 3mm
    for cand in pbb:
        is_overlap = False
        for appended in pbb_cand_list:
            minimum_dist = 3
            dist = math.sqrt(
                math.pow(appended[1] - cand[1], 2) + math.pow(appended[2] - cand[2], 2) + math.pow(
                    appended[3] - cand[3], 2))
            if (dist < minimum_dist):
                is_overlap = True
                break;

        if not is_overlap:
            pbb_cand_list.append(cand)

    pbb_cand_list = np.array(pbb_cand_list)
    pbb_cand_list_nms = nms(pbb_cand_list, 0.3)

    # print (name)
    # print (lbb)
    world_pbb = convert_prob(pbb_cand_list_nms)
    # print (world_pbb)
    print("label", len(lbb))
    print("z_pos   y_pos   x_pos   size")
    for i in range(len(lbb)):
        for j in range(len(lbb[i])):
            print(round(lbb[i][j], 2), end='\t')
        print()
    print("candidate", len(world_pbb))
    print("prob    z_pos   y_pos   x_pos   size")
    for i in range(len(world_pbb)):
        for j in range(len(world_pbb[i])):
            print(round(world_pbb[i][j], 2), end='\t')
        print()
    total_label += len(lbb)
    total_candi += len(world_pbb)

    return lbb, world_pbb

def draw_nodule_rect(lbb, world_pbb, img_arr):
    for i in range(len(lbb)):
        label = lbb[i]
        # label = np.ceil(label)
        r = (label[3] / 2) * 1.3
        top_left = (max(int(math.ceil(label[2] - r)), 0),
                    max(int(math.ceil(label[1] - r)), 0))
        bottom_right = (min(int(math.ceil(label[2] + r)), np.shape(img_arr)[1]),
                        min(int(math.ceil(label[1] + r)), np.shape(img_arr)[2]))
        z_range = [max(int(math.ceil(label[0] - r)), 0),
                   min(int(math.ceil(label[0] + r)), np.shape(img_arr)[0])]
        for j in range(z_range[0], z_range[1]):
            cv2.rectangle(img_arr[j], top_left, bottom_right, (0, 255, 0), 1)

    for i in range(len(world_pbb)):
        candidate = world_pbb[i]
        r = (candidate[4] / 2) * 1.3

        top_left = (max(int(math.ceil(candidate[3] - r)), 0),
                    max(int(math.ceil(candidate[2] - r)), 0))
        text_top_left = (max(int(math.ceil(candidate[3] - r)) - 1, 0),
                            max(int(math.ceil(candidate[2] - r)) - 1, 0))
        bottom_right = (min(int(math.ceil(candidate[3] + r)), np.shape(img_arr)[1]),
                        min(int(math.ceil(candidate[2] + r)), np.shape(img_arr)[2]))
        z_range = [max(int(math.ceil(candidate[1] - r)), 0),
                   min(int(math.ceil(candidate[1] + r)), np.shape(img_arr)[0])]

        font = cv2.FONT_HERSHEY_SIMPLEX
        for j in range(z_range[0], z_range[1]):
            cv2.rectangle(img_arr[j], top_left, bottom_right, (255, 0, 0), 1)
            #cv2.putText(img_arr[j], "c" + str(i) + "_" +str(round(candidate[0], 2)), top_left, font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_arr[j], "c" + str(i), text_top_left, font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


def crop_all(target, img_arr, crop_size = 48):
    target = np.copy(target)

    start = []
    for i in range(3):
        start.append(int(round(target[i])) - int(crop_size / 2))

    pad = []
    pad.append([0, 0])
    for i in range(3):
        leftpad = max(0, -start[i])
        rightpad = max(0, start[i] + crop_size - img_arr.shape[i + 1])
        pad.append([leftpad, rightpad])
    crop = img_arr[:,
           max(start[0], 0):min(start[0] + crop_size, img_arr.shape[1]),
           max(start[1], 0):min(start[1] + crop_size, img_arr.shape[2]),
           max(start[2], 0):min(start[2] + crop_size, img_arr.shape[3])]

    crop = np.pad(crop, pad, 'constant', constant_values=0)

    for i in range(3):
        target[i] = target[i] - start[i]

    return crop, target

def crop_nodule_arr_2ch(target, img_arr, crop_size = 48):

    img_size = [crop_size, crop_size, crop_size]
    crop_img, target = crop_all(target, img_arr, crop_size)
    imgs = np.squeeze(crop_img, axis=0)

    z = int(target[0])
    y = int(target[1])
    x = int(target[2])
    print (z, y, x)
    # z = 24
    # y = 24
    # x = 24

    nodule_size = int(target[3])
    margin = max(7, nodule_size * 0.4)
    radius = int((nodule_size + margin) / 2)

    s_z_pad = 0
    e_z_pad = 0
    s_y_pad = 0
    e_y_pad = 0
    s_x_pad = 0
    e_x_pad = 0

    s_z = max(0, z - radius)
    if (s_z == 0):
        s_z_pad = -(z - radius)

    e_z = min(np.shape(imgs)[0], z + radius)
    if (e_z == np.shape(imgs)[0]):
        e_z_pad = (z + radius) - np.shape(imgs)[0]

    s_y = max(0, y - radius)
    if (s_y == 0):
        s_y_pad = -(y - radius)

    e_y = min(np.shape(imgs)[1], y + radius)
    if (e_y == np.shape(imgs)[1]):
        e_y_pad = (y + radius) - np.shape(imgs)[1]

    s_x = max(0, x - radius)
    if (s_x == 0):
        s_x_pad = -(x - radius)

    e_x = min(np.shape(imgs)[2], x + radius)
    if (e_x == np.shape(imgs)[2]):
        e_x_pad = (x + radius) - np.shape(imgs)[2]

    # print (s_x, e_x, s_y, e_y, s_z, e_z)
    # print (np.shape(img_arr[s_z:e_z, s_y:e_y, s_x:e_x]))
    nodule_img = imgs[s_z:e_z, s_y:e_y, s_x:e_x]
    nodule_img = np.pad(nodule_img, [[s_z_pad, e_z_pad], [s_y_pad, e_y_pad], [s_x_pad, e_x_pad]], 'constant',
                        constant_values=0)

    imgpad_size = [img_size[0] - np.shape(nodule_img)[0],
                   img_size[1] - np.shape(nodule_img)[1],
                   img_size[2] - np.shape(nodule_img)[2]]
    imgpad = []
    imgpad_left = [int(imgpad_size[0] / 2),
                   int(imgpad_size[1] / 2),
                   int(imgpad_size[2] / 2)]
    imgpad_right = [int(imgpad_size[0] / 2),
                    int(imgpad_size[1] / 2),
                    int(imgpad_size[2] / 2)]

    for i in range(3):
        if (imgpad_size[i] % 2 != 0):

            rand = np.random.randint(2)
            if rand == 0:
                imgpad.append([imgpad_left[i], imgpad_right[i] + 1])
            else:
                imgpad.append([imgpad_left[i] + 1, imgpad_right[i]])
        else:
            imgpad.append([imgpad_left[i], imgpad_right[i]])

    padding_crop = np.pad(nodule_img, imgpad, 'constant', constant_values=0)

    padding_crop = np.expand_dims(padding_crop, axis=0)

    crop = np.concatenate((padding_crop, crop_img))
    crop = (crop.astype(np.float32) - 128) / 128

    return torch.from_numpy(crop), crop

def predict_attribute(attribute_net, crop_img):
    attribute_net.eval()
    crop_img = Variable(crop_img.cuda(async=True), volatile=True)
    output = attribute_net(crop_img)
    return output

