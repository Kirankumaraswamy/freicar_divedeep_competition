#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes
from utils import postprocess, STANDARD_COLORS, standard_to_bgr
import argparse
import os
import torch
import numpy as np
from torchvision import transforms
from utils import display

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', type=str, default="freicar_1", help='name of the agent')
ap.add_argument('-w', '--weights', type=str, default="/home/freicar/freicar_ws/src/Freicar_DiveDeep/02-01-object-detection-exercise/efficientdet-d0_20_30555.pth", help='/path/to/weights')
args = ap.parse_args()


compound_coef = 0  # whether you use D-0, D-1, ... etc
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.5
iou_threshold = 0.2

use_cuda = False
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

        print("Ros node initiated")
        rospy.init_node('Image_bbs', anonymous=True)
        self.pub = rospy.Publisher('predicted_bbs', Float32MultiArray, queue_size=1)
        self.r = rospy.Rate(1)  # 1hz
        self.subscriber = rospy.Subscriber("/"+name+"/sim/camera/rgb/front/image", Image, self.callback)

    def publish_bbs_to_ros(self, bbs):
        if not rospy.is_shutdown():
            msg = Float32MultiArray()
            msg.data = bbs.reshape([4 * len(bbs)])
            msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            # dim[0] is the vertical dimension of your matrix
            msg.layout.dim[0].label = "bbs"
            msg.layout.dim[0].size = len(bbs)
            # dim[1] is the horizontal dimension of your matrix
            msg.layout.dim[1].label = "samples"
            msg.layout.dim[1].size = 4
            self.pub.publish(msg)
            self.r.sleep()

    def callback(self, msg):
        model = self.model
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, 'rgb8')
        image = loadRGB(image, size=(640, 360), pad_info=(0, 0, 12, 12))
        image = torch.unsqueeze(image, dim=0).float()
        if use_cuda:
            image = image.cuda()

        with torch.no_grad():
            features, regression, classification, anchors = model(image)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        predictions = postprocess(image, anchors, regression, classification,
                                  regressBoxes, clipBoxes, threshold, iou_threshold)

        imgs = image.permute(0, 2, 3, 1).cpu().numpy()

        #display(predictions, imgs, imshow=True, imwrite=False, obj_list=obj_list)
        self.publish_bbs_to_ros(predictions[0]['rois'])

if __name__ == '__main__':

    bbx_predictor = PerceptionNode(args.name, args.weights)
    image_subscriber = rospy.Subscriber("/" + args.name + "/sim/camera/rgb/front/image", Image, bbx_predictor.callback)
    rospy.spin()