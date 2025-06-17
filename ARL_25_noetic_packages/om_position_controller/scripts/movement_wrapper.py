import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
import rosbag
from tf.transformations import quaternion_matrix
from movement_primitives.dmp import CartesianDMP
import pickle
import os
import time
from scipy.interpolate import interp1d
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import matplotlib.pyplot as plt
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
from pick_and_place import *



