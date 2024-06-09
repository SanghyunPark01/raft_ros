#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image

class Handler:
    def __init__(self):
        rospy.init_node('test_img_pair_pub', anonymous=True)
        rospy.Subscriber("/kitti/camera_color_left/image_raw", Image, self.callbackImage)
        self._m_pub_prev_img = rospy.Publisher('/test_prev', Image, queue_size=100)
        self._m_pub_curr_img = rospy.Publisher('/test_curr', Image, queue_size=100)

        self._m_first_time = True

    def callbackImage(self, msg):
        if self._m_first_time:
            self._m_prev_img = msg
            self._m_first_time = False
            return
        self._m_pub_prev_img.publish(self._m_prev_img)
        self._m_pub_curr_img.publish(msg)
        self._m_prev_img = msg

if __name__ == "__main__":
    ros_node = Handler()

    rospy.spin()