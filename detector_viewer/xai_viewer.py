import sys
import UI_util
import numpy as np
import cv2
import time
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from xai_viewer_ui import Ui_xai_viewer

import torch
import res18_split_focal as detect_model
from torch.nn import DataParallel
from torch.backends import cudnn
from training.utils import *
from training.split_combine import SplitComb


#TODO: nodule view rescale feature add
class Main_Window(QtWidgets.QMainWindow, Ui_xai_viewer):
    def __init__(self):
        super(Main_Window,self).__init__()

        ## set path and gpu number
        self.init_openpath = '/root/ssd_data/demo/'
        self.label_dirpath = '/root/ssd_data/luna_segment_attribute/'
        self.detect_resume = './detector.ckpt'
        self.gpu = '1'

        self.setupUi(self)
        self.actionOpen.triggered.connect(self.open)
        self.next_button.clicked.connect(self.next_slide)
        self.prev_button.clicked.connect(self.prev_slide)
        self.detect_button.clicked.connect(self.detect)
        self.horizontalScrollBar.valueChanged.connect(self.scroll_slide)
        self.listView.clicked.connect(self.click_nodule_list)

        self.resolution = np.array([1,1,1])
        self.slice_index = 0
        self.slice_num = 0
        self.slice_width = 0
        self.slice_height = 0

        self.detect_net, self.split_comber, self.get_pbb \
            = self.init_net()
        self.stride = 4
        self.n_per_run = 1
        self.detect_progressBar.setValue(0)
        self.fileopen_progressBar.setValue(0)

        self.file_dialog = QtWidgets.QFileDialog(directory=self.init_openpath)
        self.file_dialog.setNameFilters(["mhd files (*.mhd)", "Images (*.png *.jpg)", "All Files (*.*)"])
        self.file_dialog.selectNameFilter("mhd files (*.mhd)")

    def keyPressEvent(self, qKeyEvent):
        print(qKeyEvent.key())
        if qKeyEvent.key() == QtCore.Qt.Key_Z:
            print('Key_Left')
            self.prev_slide()
        elif qKeyEvent.key() == QtCore.Qt.Key_X:
            print('Key_Right')
            self.next_slide()
        #else:
        #    super().keyPressEvent(qKeyEvent)

    def init_net(self):
        torch.manual_seed(0)
        torch.cuda.set_device(0)

        #model = import_module(self.model)
        detect_config, detect_net, _, get_pbb = detect_model.get_model()

        detect_checkpoint = torch.load(self.detect_resume)
        detect_net.load_state_dict(detect_checkpoint['state_dict'])


        n_gpu = setgpu(self.gpu)

        detect_net = detect_net.cuda()
        #loss = loss.cuda()
        cudnn.benchmark = True
        detect_net = DataParallel(detect_net)

        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen, detect_config['max_stride'], detect_config['stride'], margin, detect_config['pad_value'])

        print ("init_net complete")
        return detect_net, split_comber, get_pbb

    def update_slide(self):
        img = np.array(self.slice_arr[self.slice_index], dtype=np.uint8)

        image = QtGui.QImage(img, self.slice_width, self.slice_height, self.slice_width * 3,
                             QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(image)
        self.slide_show_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slide_show_label.setPixmap(pixmap.scaled(791, 481, QtCore.Qt.KeepAspectRatio))
        self.slide_view_label.setText("Slide View " + str(self.slice_index) + "/" + str(self.slice_num - 1))

    def update_slidebar(self):
        self.horizontalScrollBar.blockSignals(True)
        self.horizontalScrollBar.setValue(self.slice_index)
        self.horizontalScrollBar.blockSignals(False)

    def click_nodule_list(self, QModelIndex):
        print ("click_nodule_list", QModelIndex.row())
        idx = QModelIndex.row()
        gt_num = 0

        for i in range(len(self.lbb)):
            if (self.lbb[i][3] != 0):
                gt_num += 1

        cand_num = len(self.world_pbb)

        if (idx > gt_num - 1):
            cand_idx = idx - gt_num
            if (int(round(self.world_pbb[cand_idx][1])) < 0):
                self.slice_index = 0
            elif (int(round(self.world_pbb[cand_idx][1])) > (self.slice_num - 1)):
                self.slice_index = self.slice_num - 1
            else:
                self.slice_index = int(round(self.world_pbb[cand_idx][1]))

        else:
            gt_idx = idx
            self.slice_index = int(round(self.lbb[gt_idx][0]))

        self.update_slide()
        self.update_slidebar()

    def detect(self):

        if (self.slice_num <= 0):
            return 0

        s = time.time()
        data, coord2, nzhw = UI_util.split_data(np.expand_dims(self.sliceim_re, axis=0),
                                                self.stride, self.split_comber)

        self.detect_progressBar.setValue(10)
        self.gt_path = self.label_dirpath + self.pt_num + '_label.npy'
        labels = np.load(self.gt_path)

        e = time.time()

        self.lbb, self.world_pbb = UI_util.predict_nodule(self.detect_net, data, coord2, nzhw, labels,
                               self.n_per_run, self.split_comber, self.get_pbb, self.detect_progressBar)

        nodule_items = []
        for i in range(len(self.lbb)):
            if self.lbb[i][3] != 0:
                nodule_items.append('gt_' + str(i))

        for i in range(len(self.world_pbb)):
            nodule_items.append('cand_' + str(i) + ' ' + str(round(self.world_pbb[i][0], 2)))

        model = QtGui.QStandardItemModel()
        for nodule in nodule_items:
            model.appendRow(QtGui.QStandardItem(nodule))
        self.listView.setModel(model)

        print('elapsed time is %3.2f seconds' % (e - s))
        UI_util.draw_nodule_rect(self.lbb, self.world_pbb, self.slice_arr)

        # attrbute_list = []
        # for i in range(len(self.world_pbb)):
        #     print (self.world_pbb[i][1:])
        #     print (np.shape(self.sliceim_re))
        #     crop_img, _ = UI_util.crop_nodule_arr_2ch(self.world_pbb[i][1:], np.expand_dims(self.sliceim_re, axis=0))
        #     output = UI_util.predict_attribute(self.attribute_net, crop_img.unsqueeze(0))
        #     print (output.cpu().data.numpy())
        #     attrbute_list.append(output.cpu().data.numpy())


            #print ("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test1" + str(i) + ".png")
            #print ("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test2" + str(i) + ".png")
            #cv2.imwrite("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test1" + str(i) + ".png", crop[0][24])
            #cv2.imwrite("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test2" + str(i) + ".png", crop[1][24])

        # self.print_nodule_attribute(attrbute_list)

        self.detect_progressBar.setValue(100)
        #assert False

        self.update_slide()

    def open(self):
        #TODO: file type check
        self.file_dialog.exec_()
        fileName = self.file_dialog.selectedFiles()

        print("open ",fileName)

        if (fileName[0] == ''):
            return 0

        self.pt_num = fileName[0].split('/')[-1].split('.mhd')[0]
        self.detect_progressBar.setValue(0)
        self.fileopen_progressBar.setValue(0)
        # self.tableWidget.setRowCount(0)
        # self.tableWidget.setColumnCount(0)
        self.file_name.setText(fileName[0] + " opening ...")

        model = QtGui.QStandardItemModel()
        self.listView.setModel(model)

        sliceim, origin, spacing, isflip = UI_util.load_itk_image(fileName[0])

        self.fileopen_progressBar.setValue(10)

        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = UI_util.lumTrans(sliceim)


        self.sliceim_re, _ = UI_util.resample(sliceim, spacing, self.resolution, self.fileopen_progressBar, order=1)

        self.fileopen_progressBar.setValue(45)

        self.slice_arr = np.zeros((np.shape(self.sliceim_re)[0], np.shape(self.sliceim_re)[1], np.shape(self.sliceim_re)[2], 3))

        self.slice_num = np.shape(self.sliceim_re)[0]
        self.slice_height = np.shape(self.sliceim_re)[1]
        self.slice_width = np.shape(self.sliceim_re)[2]

        for i in range(len(self.sliceim_re)):
            self.slice_arr[i] = cv2.cvtColor(self.sliceim_re[i], 8)
            self.fileopen_progressBar.setValue(45 + (45/len(self.sliceim_re))*(i+1))

        print ("finish convert")
        self.slice_index = int(self.slice_num/2)
        img = np.array(self.slice_arr[self.slice_index], dtype=np.uint8)

        image = QtGui.QImage(img, self.slice_width, self.slice_height, self.slice_width*3, QtGui.QImage.Format_RGB888)

        self.update_slide()

        self.file_name.setText(fileName[0] + " open completed ...")

        self.horizontalScrollBar.setMaximum(self.slice_num - 1)
        self.horizontalScrollBar.setMinimum(0)

        self.update_slidebar()
        self.fileopen_progressBar.setValue(100)

    def next_slide(self):
        if self.slice_index < self.slice_num - 1:
            self.slice_index += 1

        if (self.slice_num > 0):
            self.update_slide()
            self.update_slidebar()

    def prev_slide(self):
        if self.slice_index > 0:
            self.slice_index -= 1

        if (self.slice_num > 0):
            self.update_slide()
            self.update_slidebar()

    def scroll_slide(self):
        if (self.slice_num > 0):
            self.slice_index = self.horizontalScrollBar.value()
            self.update_slide()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("XAI Viewer")

    window = Main_Window()
    window.show()
    app.exec_()