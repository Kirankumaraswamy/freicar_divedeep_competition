#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose
import tf

from birdsEyeT import birdseyeTransformer
from model.fast_scnn_model import Fast_SCNN
from dataset_helper import color_coder

parser = argparse.ArgumentParser(description='Segmentation and Regression publishing')
parser.add_argument('--load_model',
                    default='recent_model_19.pth.tar',
                    type=str, metavar='PATH',
                    help='path to model (default: none)')
args = None
model = Fast_SCNN(3, 4)
model = model.cuda()


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
        self.pub_seg = rospy.Publisher('semantic_segmentation', Image, queue_size=1)
        self.pub_lane = rospy.Publisher('lane_detection', Image, queue_size=1)
        self.r = rospy.Rate(1)  # 1hz
        self.subscriber = rospy.Subscriber("/freicar_1/sim/camera/rgb/front/image", Image, self.callback)
        self.hom_conv = birdseyeTransformer('freicar_homography.yaml', 3, 3, 200, 2)

    def callback(self, msg):
        global id
        num_of_points = 1000
        rospy.loginfo('Image received...')
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.image = loadRGB(self.image, size=(640, 360), pad_info=(0, 0, 12, 12))
        self.image = torch.unsqueeze(self.image, dim=0)
        seg_pred, lane_pred = self.get_seg_lane_pred(self.image.cuda())
        lane_pred = lane_pred[:, :, 12:372, :]  # Unpadding from top and bottom
        lane_pred = TensorImage1ToCV(lane_pred)
        lreg_norm = cv2.normalize(lane_pred, None, alpha=0.00, beta=1.00, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        _, lreg_threshold = cv2.threshold(lreg_norm, 0.80, 1.00, cv2.THRESH_TOZERO)
        bev = self.hom_conv.birdseye(lreg_threshold)
        bev = bev * 255
        bev = bev.astype(np.uint8)
        lane_img = self.publish_seg_lane(seg_pred, bev)
        lane_img.header.stamp = msg.header.stamp
        np_points = np.argwhere(bev > 200)
        if len(np_points) > num_of_points:
            print(np_points.shape)
            ids = np.random.choice(np_points[:, 0], num_of_points)
            np_points = np_points[ids, :]

        id = 0
        if not rospy.is_shutdown():
            marker_array = MarkerArray()
            count = 0
            for pt in np_points:
                marker = Marker()
                pose = Pose()
                point = Point()
                point.y = pt[1] / 200.0 - 1.5
                point.x = pt[0] / 200.0 - 0.2
                point.z = 0.0
                pose.position = point
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                marker.id = count
                marker.pose = pose
                marker.color.g = 0.0
                marker.color.r = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.header.frame_id = "freicar_1/base_link"
                marker.header.stamp = rospy.Time.now()
                marker_array.markers.append(marker)
                count += 1

            pub = rospy.Publisher('/lreg', MarkerArray, queue_size=10)
            pub.publish(marker_array)


    def get_seg_lane_pred(self, image):
        model.eval()
        color_conv = color_coder.ColorCoder()
        with torch.no_grad():
            # load model weights file from disk
            seg_pred, lane_pred = model(image)
            seg_pred = color_conv.color_code_labels(seg_pred)
        return seg_pred, lane_pred


    def publish_seg_lane(self, seg_pred, lane_pred):
        if not rospy.is_shutdown():
            bridge = CvBridge()
            seg_pred = bridge.cv2_to_imgmsg(seg_pred)
            lane_pred = bridge.cv2_to_imgmsg(lane_pred)
            self.pub_seg.publish(seg_pred)
            rospy.loginfo('Segmentation Published...')
            self.pub_lane.publish(lane_pred)
            rospy.loginfo('Lane Regression Published...')
            self.r.sleep()
            return lane_pred


def main():
    global args
    args = parser.parse_args()
    if args.load_model:
        if os.path.isfile(args.load_model):
            print("=> loading model '{}'".format(args.load_model))
            model.load_state_dict(torch.load(args.load_model)['state_dict'])
        else:
            print("=> no model found at '{}'".format(args.load_model))
    rospy.init_node('lane_detector', anonymous=True)
    print("Ros node initiated")
    lane = LaneDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
