#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
import random

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import os
import numpy as np
import torch
import sys
import gc

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', type=str, default="greatteam", help='name of the agent')
ap.add_argument('-w', '--weights', type=str, default="recent_model_19.pth.tar", help='/path/to/weights')
ap.add_argument('-p', '--folder_path', type=str, default="/home/freicar/freicar_ws/src/ros_code/02-02-semantic-segmentation-exercise", help='path to directory')
args = ap.parse_args()

sys.path.append(args.folder_path)

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose
import tf

from birdsEyeT import birdseyeTransformer
from model.fast_scnn_model import Fast_SCNN
from dataset_helper import color_coder

use_cuda = True


def TensorImage1ToCV(data):
    cv = data.cpu().data.numpy().squeeze()
    return cv


def loadRGB(image, size=None, pad_info=None):
    rgb_im = [image]
    if size is not None:
        rgb_im = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in rgb_im]
    rgb_im = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in rgb_im]
    rgb_im = [(torch.from_numpy(im)).permute(2, 0, 1) / 255. for im in rgb_im]
    if pad_info is not None:
        rgb_im = [torch.nn.functional.pad(d, pad_info, 'constant', 0) for d in rgb_im]
    return rgb_im[0]


def visJetColorCoding(img):
    img = img
    color_img = np.zeros(img.shape, dtype=(img.astype(np.uint8)).dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    return color_img


class LaneDetector(object):
    def __init__(self):

        print("lane detector node initiated")
        rospy.init_node('Lane_regression', anonymous=True)

        self.agent_name = args.name #rospy.get_param("/agent_name")
        self.pub_seg = rospy.Publisher('semantic_segmentation', Image, queue_size=1)
        self.pub_lane = rospy.Publisher(self.agent_name+"/sim/camera/rgb/front/reg_bev", Image, queue_size=1)
        #self.rate = rospy.Rate(1)  # 1hz
        self.subscriber = rospy.Subscriber(self.agent_name + "/sim/camera/rgb/front/image", Image, self.callback, queue_size=1)
        self.hom_conv = birdseyeTransformer('freicar_homography.yaml', 3, 3, 200, 2)
        self.model = Fast_SCNN(3, 4)
        self.model.requires_grad_(False)
        self.model.eval()

        if use_cuda:
            self.model = self.model.cuda()

        if args.weights:
            self.model.load_state_dict(torch.load(args.weights, map_location='cpu')['state_dict'])

        self.r = rospy.Rate(250)


    def callback(self, msg):
        rospy.loginfo('Image received...')
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, 'rgb8')
        image = loadRGB(image, size=(640, 360), pad_info=(0, 0, 12, 12))
        image1 = torch.unsqueeze(image, dim=0)

        torch.cuda.empty_cache()
        if use_cuda:
            image1 = image1.cuda()

        with torch.no_grad():
            seg_pred, lane_pred = self.model(image1)

        del image1
        gc.collect


        lane_pred = lane_pred[:, :, 12:372, :]  # Unpadding from top and bottom
        lane_pred = TensorImage1ToCV(lane_pred)
        lreg_norm = cv2.normalize(lane_pred, None, alpha=0.00, beta=1.00, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        _, lreg_threshold = cv2.threshold(lreg_norm, 0.80, 1.00, cv2.THRESH_TOZERO)
        #lreg_threshold = cv2.rotate(lreg_threshold, cv2.cv2.ROTATE_90_CLOCKWISE)


        if len(np.where(lreg_threshold > 0.8)[0]) > 0:
            bev = self.hom_conv.birdseye(lreg_threshold)
            #lreg_bev = cv2.flip(bev, -1)
            #bev = self.hom_conv.reverse_birdseye(lreg_threshold, (600, 384)) frei


            #converting from 0 to 1 - > 0 to 255
            bev = bev * 255

            bev = bev.astype(np.uint8)
            bev = CvBridge().cv2_to_imgmsg(bev, 'mono8')
            bev.header.stamp = msg.header.stamp
            #bev.header.stamp = rospy.Time.now()
            lane_img = self.publish_seg_lane(seg_pred, bev)


    def publish_seg_lane(self, seg_pred, lane_pred):
        if not rospy.is_shutdown():
            self.pub_lane.publish(lane_pred)
            rospy.loginfo('Lane Regression Published...')
            #self.rate.sleep()
            return lane_pred



if __name__ == '__main__':
    lane = LaneDetector()
    rospy.spin()
