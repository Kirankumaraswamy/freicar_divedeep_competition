#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PerceptionNode(object):
    def __init__(self):
        print("Ros node initiated")
        rospy.init_node('Image_bbs', anonymous=True)
        self.pub = rospy.Publisher('predicted_bbs', Float32MultiArray, queue_size=1)
        self.r = rospy.Rate(1)  # 1hz
        self.subscriber = rospy.Subscriber("/freicar_anyname/sim/camera/rgb/front/image", Image, self.callback)


    def callback(self, msg):
        rospy.loginfo('Image received...')
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow("image", self.image)
        cv2.waitKey(0)


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
