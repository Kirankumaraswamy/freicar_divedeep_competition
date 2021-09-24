#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters


import argparse
import os
import torch
import numpy as np
from torchvision import transforms
import gc

import sys

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', type=str, default="greatteam", help='name of the agent')
ap.add_argument('-w', '--weights', type=str, default="/home/freicar/freicar_ws/src/ros_code/02-01-object-detection-exercise/efficientdet-d0_20_30555.pth", help='/path/to/weights')
ap.add_argument('-p', '--folder_path', type=str, default="/home/freicar/freicar_ws/src/ros_code/02-01-object-detection-exercise", help='path to directory')
args = ap.parse_args()

sys.path.append(args.folder_path)
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes
from utils import postprocess, STANDARD_COLORS, standard_to_bgr
from utils import display

compound_coef = 0  # whether you use D-0, D-1, ... etc
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.8
iou_threshold = 0.2

use_cuda = True
obj_list = ['freicar']

def loadRGB(image, size=None, pad_info=None):
    rgb_im = [image]
    if size is not None:
        rgb_im = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in rgb_im]
    rgb_im = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in rgb_im]
    rgb_im = [(torch.from_numpy(im)).permute(2, 0, 1) / 255. for im in rgb_im]
    if pad_info is not None:
        rgb_im = [torch.nn.functional.pad(d, pad_info, 'constant', 0) for d in rgb_im]
    return rgb_im[0]

class PerceptionNode(object):
    def __init__(self, name="freicar_1", weights=None):
        self.model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(obj_list),
                                     ratios=anchor_ratios,
                                     scales=anchor_scales)

        if args.weights:
            self.model.load_state_dict(torch.load(args.weights, map_location='cpu'))

        self.model.requires_grad_(False)
        self.model.eval()

        if use_cuda:
            self.model = self.model.cuda()

        print("Ros node initiated")
        rospy.init_node('Image_bbs', anonymous=True)
        self.pub = rospy.Publisher(name+'/predicted_bbs', Float32MultiArray, queue_size=1)
        self.r = rospy.Rate(250)

        #self.subscriber = rospy.Subscriber("/"+name+"/sim/camera/rgb/front/image", Image, self.rgb_callback)
        #self.subscriber_depth = rospy.Subscriber("/"+name+"/sim/camera/depth/front/image_float", Image, self.depth_callback)

        self.count = 0

        image_subscriber = message_filters.Subscriber("/"+name+"/sim/camera/rgb/front/image", Image, queue_size=1)
        depth_subscriber = message_filters.Subscriber("/"+name+"/sim/camera/depth/front/image_float", Image, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_subscriber, depth_subscriber], 2, 0.1)
        self.ts.registerCallback(self.callback)

    def publish_bbs_to_ros(self, bbs):
        if not rospy.is_shutdown():
            msg = Float32MultiArray()
            msg.data = bbs.reshape([6 * len(bbs)])
            msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            # dim[0] is the vertical dimension of your matrix
            msg.layout.dim[0].label = "bbs"
            msg.layout.dim[0].size = len(bbs)
            # dim[1] is the horizontal dimension of your matrix
            msg.layout.dim[1].label = "samples"
            msg.layout.dim[1].size = 6
            self.pub.publish(msg)
            self.r.sleep()


    def callback(self, msg=None, depth_image=None):
        #print("call back...." + str(self.count))
        self.count += 1

        if msg != None and depth_image != None:

            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, 'rgb8')
            image = loadRGB(image, size=(640, 360), pad_info=(0, 0, 12, 12))
            image1 = torch.unsqueeze(image, dim=0).float()
            if use_cuda:
                image1 = image1.cuda()

            with torch.no_grad():
                features, regression, classification, anchors = self.model(image1)



            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            predictions = postprocess(image1, anchors, regression, classification,
                                      regressBoxes, clipBoxes, threshold, iou_threshold)

            #imgs = image.permute(0, 2, 3, 1).cpu().numpy()
            del image1
            gc.collect

            depth_image = bridge.imgmsg_to_cv2(depth_image, "32FC1")
            #cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
            depth_image = cv2.resize(depth_image, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

            depth_array = np.array(depth_image, dtype=np.float32)

            cx = 596.6277465820312 * .5;
            cy = 337.6268615722656 * .5;
            fx = 725.3607788085938/2;
            fy = 725.3607788085938/2;

            intrinsic = np.array([[ fx, 0, cx],[0, fx, cy],[0, 0, 1]])
            intrinsic_inv = np.linalg.inv(intrinsic)

            new_bbs_array = []
            for i in range(len(predictions[0]['rois'])):
                top_x = int(predictions[0]['rois'][i][0]);
                top_y = int(predictions[0]['rois'][i][1]) -12 ;
                bottom_x= int(predictions[0]['rois'][i][2]);
                bottom_y = int(predictions[0]['rois'][i][3])-12;
                width = bottom_x-top_x;
                height = bottom_y-top_y;
                x_center = int(top_x + width/2);
                y_center = int(top_y + height/2);

                # in numpy the values are reversed
                row1 = top_y
                row2 = bottom_y
                col1 = top_x
                col2 = bottom_x

                row_center = int((row1+row2)/2)
                col_center = int((col1+col2)/2)

                center_depth = depth_array[row_center][col_center]
                point_depth = center_depth
                point = np.array([x_center, y_center, 1])

                camera_point = point_depth * intrinsic_inv.dot(point)
                point_x = camera_point[2]
                point_y = -camera_point[0]

                new_bbs_array.append([top_x, top_y, bottom_x, bottom_y, point_x, point_y])


            #display(predictions, imgs, imshow=True, imwrite=False, obj_list=obj_list)
            print(new_bbs_array)
            self.publish_bbs_to_ros(np.array(new_bbs_array, dtype=float))




if __name__ == '__main__':

    bbx_predictor = PerceptionNode(args.name, args.weights)
    rospy.spin()
