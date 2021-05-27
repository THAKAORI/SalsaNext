#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import imp
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax
import tasks.semantic.modules.adf as adf

import open3d as o3d

def keep_variance_fn(x):
    return x + 1e-3

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true


class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNext" + suffix)


class DataAnalysis():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None,uncertainty=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.uncertainty = uncertainty

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/data_extractor.py')
        self.parser = parserModule.DataExtractor(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=16,
                                          gt=True,
                                          shuffle_train=True)

    def data_analysis(self):

        train_set = self.parser.train_dataset

        train_set_size = len(train_set)

        segment_angle_average = 0.0
        segment_angle_num = 0.0

        for i in range(train_set_size):
            proj_range, proj_segment_angle, proj_xyz, proj_remission, proj_mask, proj_labels, points = train_set[i]
            segment_angle_valid = proj_segment_angle[proj_segment_angle > 0]
            segment_angle_average += torch.sum(segment_angle_valid).item() / segment_angle_valid.shape[0]
            segment_angle_num += 1.0
            if(i % 100 == 0):
                print(segment_angle_average / segment_angle_num)
        
        segment_angle_average /= segment_angle_num
        print("average segment angle:", segment_angle_average)

        segment_angle_std = 0.0
        segment_angle_num = 0.0

        for i in range(train_set_size):
            proj_range, proj_segment_angle, proj_xyz, proj_remission, proj_mask, proj_labels = train_set[i]
            segment_angle_valid = proj_segment_angle[proj_segment_angle > 0]
            segment_angle_std += torch.sum((segment_angle_valid - segment_angle_average) ** 2).item() / segment_angle_valid.shape[0]
            segment_angle_num += 1.0
            if(i % 100 == 0):
                print(segment_angle_std / segment_angle_num)
        
        segment_angle_std /= segment_angle_num
        segment_angle_std = np.sqrt(segment_angle_std)

        print("average segment std:", segment_angle_std)

    def datao3d(self):
        train_set = self.parser.train_dataset

        train_set_size = len(train_set)

        proj_range, proj_segment_angle, proj_xyz, proj_remission, proj_mask, proj_labels, points0, scan_file0, pose0 = train_set[0]
        proj_range, proj_segment_angle, proj_xyz, proj_remission, proj_mask, proj_labels, points1, scan_file1, pose1 = train_set[1]

        print(scan_file0)
        print(pose0)
        print(scan_file1)
        print(pose1)

        N0, _ = points0.shape

        colors0 = np.tile([1,0,0], (N0, 1))

        N1, _ = points1.shape

        colors1 = np.tile([0,1,0], (N1, 1))

        points = np.vstack((points0, points1))
        colors = np.vstack((colors0, colors1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    
    def getpointspose(self, i):
        train_set = self.parser.valid_dataset

        proj_range, proj_segment_angle, proj_xyz, proj_remission, proj_mask, proj_labels, points, scan_file, pose, sem_label = train_set[i]

        return points, pose, scan_file, sem_label
