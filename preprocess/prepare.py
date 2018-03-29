import os
import shutil
import numpy as np
from config_training import config


#from scipy.io import loadmat
import numpy as np
#import h5py
#import pandas
#import scipy
from scipy.ndimage.interpolation import zoom
#from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
#from multiprocessing import Pool
#from functools import partial
import sys
import math
sys.path.append('../preprocessing')
#from step1 import step1_python_luna
#import warnings
import glob
from bs4 import BeautifulSoup

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
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
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def savenpy_luna_attribute(xml_path, annos, filelist, luna_data, savepath, candidate_annos, abbrevs, readlist):
    islabel = True
    isClean = True
    isCandidate = True
    isAttribute = True
    resolution = np.array([1, 1, 1])
    namelist = list(abbrevs[:, 1])
    ids = abbrevs[:, 0]

    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return -1
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    if patient_id in namelist:
        name = ids[namelist.index(patient_id)]

        name = str(name)
        if len(name) < 3:
            for i in range(3 - len(name)):
                name = '0' + name
        print (name)

        if name in readlist:
            print ("overlap", name)
            return -1

        #print (id, patient_id)
        this_annos = np.copy(annos[annos[:, 0] == int(name)])
        if isClean:

            sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name + '.mhd'))
            ori_sliceim_shape_yx = sliceim.shape[1:3]
            if isflip:
                sliceim = sliceim[:, ::-1, ::-1]
                print('flip!')
            sliceim = lumTrans(sliceim)

            sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
            sliceim = sliceim1[np.newaxis, ...]
            np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)

        #make attribute_annos
        # name,pos_x, pos_y, pos_z, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety, hit_count
        if isAttribute:
            luna_annos = np.copy(this_annos)
            annos_shape = np.shape(luna_annos)
            attribute_annos = np.zeros((annos_shape[0], annos_shape[1] + 10))

            for i in range(len(luna_annos)):
                luna_annos[i][1] = (luna_annos[i][1] - origin[2]) / spacing[2]
                luna_annos[i][2] = (luna_annos[i][2] - origin[1]) / spacing[1]
                luna_annos[i][3] = (luna_annos[i][3] - origin[0]) / spacing[0]

                if isflip:
                    luna_annos[i][1] = -luna_annos[i][1]
                    luna_annos[i][2] = -luna_annos[i][2]
                attribute_annos[i] = np.concatenate((luna_annos[i], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))

            reading_sessions = xml.LidcReadMessage.find_all("readingSession")
            for reading_session in reading_sessions:
                # print("Sesion")
                nodules = reading_session.find_all("unblindedReadNodule")
                for nodule in nodules:
                    nodule_id = nodule.noduleID.text
                    rois = nodule.find_all("roi")
                    x_min = y_min = z_min = 999999
                    x_max = y_max = z_max = -999999
                    for roi in rois:
                        z_pos = float(roi.imageZposition.text)
                        z_min = min(z_min, z_pos)
                        z_max = max(z_max, z_pos)
                        edge_maps = roi.find_all("edgeMap")
                        for edge_map in edge_maps:
                            x = float(edge_map.xCoord.text)
                            y = float(edge_map.yCoord.text)
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x)
                            y_max = max(y_max, y)
                        if x_max == x_min:
                            continue
                        if y_max == y_min:
                            continue
                    x_diameter = x_max - x_min
                    x_center = x_min + x_diameter / 2
                    y_diameter = y_max - y_min
                    y_center = y_min + y_diameter / 2
                    z_diameter = z_max - z_min
                    z_center = z_min + z_diameter / 2
                    z_center -= origin[0]
                    z_center /= spacing[0]

                    if nodule.characteristics is None:
                        # print("!!!!Nodule:", nodule_id, " has no charecteristics")
                        continue
                    if nodule.characteristics.malignancy is None:
                        # print("!!!!Nodule:", nodule_id, " has no malignacy")
                        continue

                    malignacy = int(nodule.characteristics.malignancy.text)
                    sphericiy = int(nodule.characteristics.sphericity.text)
                    margin = int(nodule.characteristics.margin.text)
                    spiculation = int(nodule.characteristics.spiculation.text)
                    texture = int(nodule.characteristics.texture.text)
                    calcification = int(nodule.characteristics.calcification.text)
                    internal_structure = int(nodule.characteristics.internalStructure.text)
                    lobulation = int(nodule.characteristics.lobulation.text)
                    subtlety = int(nodule.characteristics.subtlety.text)

                    for annos in attribute_annos:
                        dist = math.sqrt(math.pow(x_center - annos[1], 2) + math.pow(y_center - annos[2], 2) + math.pow(
                            z_center - annos[3], 2))
                        if dist <= annos[4]:
                            annos[5] += malignacy
                            annos[6] += sphericiy
                            annos[7] += margin
                            annos[8] += spiculation
                            annos[9] += texture
                            annos[10] += calcification
                            annos[11] += internal_structure
                            annos[12] += lobulation
                            annos[13] += subtlety
                            annos[14] += 1

            for annos in attribute_annos:
                if (annos[14] > 0):
                    annos[5] = annos[5] / annos[14]
                    annos[6] = annos[6] / annos[14]
                    annos[7] = annos[7] / annos[14]
                    annos[8] = annos[8] / annos[14]
                    annos[9] = annos[9] / annos[14]
                    annos[10] = annos[10] / annos[14]
                    annos[11] = annos[11] / annos[14]
                    annos[12] = annos[12] / annos[14]
                    annos[13] = annos[13] / annos[14]
                else:
                    print ('no hit nodule', annos)

        if islabel:

            this_annos = np.copy(this_annos)
            label = []
            if len(this_annos) > 0:

                for c in this_annos:
                    pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                    if isflip:
                        pos[1:] = ori_sliceim_shape_yx - pos[1:]
                    label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

            label = np.array(label)
            if len(label) == 0:
                label2 = np.array([[0, 0, 0, 0]])
            else:
                label2 = np.copy(label).T
                label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
                label2[3] = label2[3] * spacing[1] / resolution[1]
                label2 = label2[:4].T

            #set voxel z,y,x pos to attibute annos
            for i in range(len(attribute_annos)):
                attribute_annos[i][1] = label2[i][0]
                attribute_annos[i][2] = label2[i][1]
                attribute_annos[i][3] = label2[i][2]

            np.save(os.path.join(savepath, name + '_label.npy'), label2)
            np.save(os.path.join(savepath, name + '_attribute.npy'), attribute_annos)

        if isCandidate:
            img_shape = sliceim.shape[1:4]
            candidate_annos = np.copy(candidate_annos[candidate_annos[:, 0] == int(name)])
            label = []
            if len(candidate_annos) > 0:

                for c in candidate_annos:
                    pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                    # print ("2 label", pos)
                    if isflip:
                        pos[1:] = ori_sliceim_shape_yx - pos[1:]
                        # print ("flip label", pos)

                    pos = pos * spacing / resolution

                    transit_val = 6
                    min_dist = 3
                    min_check = ((pos - transit_val) > 0).all()
                    max_check = (((pos + transit_val) - img_shape) < 0).all()
                    # print (min_check, max_check)
                    if (min_check and max_check):
                        label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

            label = np.array(label)
            print ("candidate len", len(candidate_annos), len(label))
            if len(label) == 0:
                label2 = np.array([[0, 0, 0, 0]])
            else:
                label2 = np.copy(label).T
                # print ("3 label", label2)
                # label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
                label2[3] = label2[3] * spacing[1] / resolution[1]
                # label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
                # print ("a label", label2)
                label2 = label2[:4].T
            np.save(os.path.join(savepath, name + '_candidate.npy'), label2)

        return name
    else:
        print ('not LUNA16 list', patient_id)
        return -1
    #name = filelist[id]
    print(id, name)

    return 1


def prepare_luna():
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    #luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'

    if not os.path.exists(finished_flag):
        print('start changing luna name')
        subsetdirs = [os.path.join(luna_raw, f) for f in os.listdir(luna_raw) if
                      f.startswith('subset') and os.path.isdir(os.path.join(luna_raw, f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

        abbrevs = np.array(pandas.read_csv(config['luna_abbr'], header=None))
        namelist = list(abbrevs[:, 1])
        ids = abbrevs[:, 0]

        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()

            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0' * (3 - len(str(id))) + str(id)
                shutil.move(os.path.join(d, f), os.path.join(luna_data, filename + f[-4:]))
                print(os.path.join(luna_data, str(id) + f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data, file), 'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0' * (3 - len(str(id))) + str(id)
                content[-1] = 'ElementDataFile = ' + filename + '.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data, file), 'w') as f:
                f.writelines(content)

    print('end changing luna name')
    f = open(finished_flag, "w+")

def preprocess_luna():
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    luna_candidate_label = config['luna_candidate_label']
    finished_flag = '.flag_preprocessluna'
    xml_path = config['lidc_xml']

    abbrevs = np.array(pandas.read_csv(config['luna_abbr'], header=None))

    print('starting preprocessing luna', os.path.exists(finished_flag))
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))
        candidate_annos = np.array(pandas.read_csv(luna_candidate_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        readlist = []
        file_no = 0
        for anno_dir in [d for d in glob.glob(xml_path + "/*") if os.path.isdir(d)]:
            xml_paths = glob.glob(anno_dir + "/*.xml")
            print(file_no, ": ", xml_path)
            for xml_path in xml_paths:
                err = savenpy_luna_attribute(xml_path=xml_path, annos=annos, filelist=filelist, luna_data=luna_data,
                                   savepath=savepath, candidate_annos=candidate_annos, abbrevs=abbrevs, readlist=readlist)
                if  (err != -1):
                    if  err not in readlist:
                        readlist.append(err)
                    file_no += 1
        print('end preprocessing luna', file_no, len(filelist),)

    f= open(finished_flag,"w+")
    
if __name__=='__main__':
    prepare_luna()
    preprocess_luna()

