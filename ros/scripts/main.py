#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from queue import Queue
import threading
from threading import Thread

import sys
import os
path__ = os.path.dirname(os.path.realpath(__file__))
path__ += "/../../"
sys.path.append(path__)
import argparse
import cv2
import glob
import numpy as np
import torch
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

class RAFT_ROS:
    def __init__(self):
        rospy.init_node('raft_ros', anonymous=True)

        rospy.Subscriber(rospy.get_param("/ROS/prev_img"), Image, self.callbackPrevImage)
        rospy.Subscriber(rospy.get_param("/ROS/curr_img"), Image, self.callbackCurrImage)
        self._m_pub_result = rospy.Publisher('/raft_result', Image, queue_size=100)

        parser = argparse.ArgumentParser()
        parser.add_argument('model')
        parser.add_argument('small')
        self.args = parser.parse_args()
        self.args.model = rospy.get_param("/RAFT/weight_path")
        self.args.small = rospy.get_param("/RAFT/is_small")
        self.args.mixed_precision = False
        self.device = rospy.get_param("/RAFT/device")
        
        self._m_que_prev_img = Queue()
        self._m_mtx_prev_que = threading.Lock()
        self._m_que_curr_img = Queue()
        self._m_mtx_curr_que = threading.Lock()
        self._m_que_pair_img = Queue()
        self._m_mtx_pair_que = threading.Lock()
        
        print("Initialize RAFT...")
        self.initModel()
        print("Initialize RAFT finish!")

    def initModel(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def callbackPrevImage(self, msg):
        self._m_mtx_prev_que.acquire()
        self._m_que_prev_img.put(msg)
        self._m_mtx_prev_que.release()

    def callbackCurrImage(self, msg):
        self._m_mtx_curr_que.acquire()
        self._m_que_curr_img.put(msg)
        self._m_mtx_curr_que.release()

    def convert2Img(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
        # cv2.waitKey(1)    
        return flo[:, :, [2,1,0]]/255.0

    def thdInference(self):
        while True:
            self.syncImg()
            self.inferRAFT()

    def syncImg(self):
        if self._m_que_curr_img.empty() or self._m_que_prev_img.empty():
            return
        
        # get curr image
        self._m_mtx_curr_que.acquire()
        curr_img = self._m_que_curr_img.get()
        self._m_mtx_curr_que.release()
        
        # get prev image
        self._m_mtx_prev_que.acquire()
        while True:
            if self._m_que_prev_img.empty():
                self._m_mtx_prev_que.release()
                return
            
            prev_img = self._m_que_prev_img.get()
            if prev_img.header.stamp.to_sec() < curr_img.header.stamp.to_sec():
                break
        self._m_mtx_prev_que.release()

        # make image pair
        cv_bridge__ = CvBridge()
        cv_prev_img = cv_bridge__.imgmsg_to_cv2(prev_img, desired_encoding='passthrough')
        cv_curr_img = cv_bridge__.imgmsg_to_cv2(curr_img, desired_encoding='passthrough')
        self._m_mtx_pair_que.acquire()
        self._m_que_pair_img.put([curr_img.header, cv_prev_img, cv_curr_img])
        self._m_mtx_pair_que.release()


    def inferRAFT(self):
        if self._m_que_pair_img.empty():
            return
        
        # get pair
        self._m_mtx_pair_que.acquire()
        img_pair = self._m_que_pair_img.get()
        self._m_mtx_pair_que.release()

        curr_ros_header = img_pair[0]
        prev_img = img_pair[1]
        curr_img = img_pair[2]

        with torch.no_grad():
            prev_img = np.array(prev_img).astype(np.uint8)
            image1 = torch.from_numpy(prev_img).permute(2, 0, 1).float().to(self.device)
            image1 = image1.unsqueeze(0)
            curr_img = np.array(curr_img).astype(np.uint8)
            image2 = torch.from_numpy(curr_img).permute(2, 0, 1).float().to(self.device)
            image2 = image2.unsqueeze(0)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

            result_img = self.convert2Img(image1, flow_up)
            result_img = (result_img * 255).astype(np.uint8)
            cv_bridge__ = CvBridge()
            curr_ros_header.frame_id = "raft_image"
            img_to_pub = cv_bridge__.cv2_to_imgmsg(result_img, encoding="passthrough", header=curr_ros_header)

            self._m_pub_result.publish(img_to_pub)

        
            

if __name__ == '__main__':
    raft_ros = RAFT_ROS()

    th = Thread(target=raft_ros.thdInference, daemon=True)
    th.start()

    rospy.spin()