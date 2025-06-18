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
from simpile_ik import *
import geometry_msgs.msg # Added for the main example
from tf.transformations import quaternion_from_euler



def get_cube_position(cube_name, timeout=5.0):
    """Get position of a cube by its TF frame name"""
    print(f"Getting {cube_name} position...")
    if not rospy.core.is_initialized():
        rospy.init_node('tf_xyz_fetcher', anonymous=True)
    
    listener = tf.TransformListener()
    try:
        print(f"Waiting for transform /world -> /{cube_name}...")
        listener.waitForTransform('/world', f'/{cube_name}', rospy.Time(0), rospy.Duration(timeout))
        
        trans, _ = listener.lookupTransform('/world', f'/{cube_name}', rospy.Time(0))
        print(f"{cube_name} position: {trans}")
        return trans
    except Exception as e:
        print(f"Error getting transform for {cube_name}: {e}")
        return None


class GazeboLinkAttacher:
    """Handles attaching and detaching links in Gazebo by calling the correct services."""
    def __init__(self):
        rospy.loginfo("Initializing GazeboLinkAttacher")
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        
        rospy.loginfo("Waiting for /link_attacher_node/attach and /link_attacher_node/detach services...")
        self.attach_srv.wait_for_service()
        self.detach_srv.wait_for_service()
        rospy.loginfo("Gazebo link attacher services are ready.")

    def attach(self, model_name_1, link_name_1, model_name_2, link_name_2):
        rospy.loginfo(f"Attaching {model_name_1}:{link_name_1} to {model_name_2}:{link_name_2}")
        req = AttachRequest()
        req.model_name_1 = model_name_1
        req.link_name_1 = link_name_1
        req.model_name_2 = model_name_2
        req.link_name_2 = link_name_2
        try:
            self.attach_srv.call(req)
            rospy.loginfo("Attachment successful.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to attach: {e}")

    def detach(self, model_name_1, link_name_1, model_name_2, link_name_2):
        rospy.loginfo(f"Detaching {model_name_1}:{link_name_1} from {model_name_2}:{link_name_2}")
        req = AttachRequest()
        req.model_name_1 = model_name_1
        req.link_name_1 = link_name_1
        req.model_name_2 = model_name_2
        req.link_name_2 = link_name_2
        try:
            self.detach_srv.call(req)
            rospy.loginfo("Detachment successful.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to detach: {e}")


class GazeboController:
    """
    A simplified controller to send direct joint commands to a robot's arm and gripper in Gazebo.
    """

    def __init__(self, arm_joint_names=None, gripper_joint_names=None,home_position=None):
        """
        Initializes the GazeboController.

        :param arm_joint_names: List of joint names for the arm.
        :param gripper_joint_names: List of joint names for the gripper.
        """
        # Initialize the ROS node if it hasn't been already.
        if not rospy.core.is_initialized():
            rospy.init_node("gazebo_controller", anonymous=True)

        # Default joint names for the Open Manipulator
        self.arm_joint_names = arm_joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = gripper_joint_names or ["gripper", "gripper_sub"]

        self.home_position = home_position if home_position is not None else [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194]


        # --- TF Listener ---
        # Initialize a TF listener to look up transformations.
        rospy.loginfo("Initializing TF listener.")
        self.listener = tf.TransformListener()
        self.base_frame = 'world' # The reference frame for movements

        # --- Publishers ---
        # Publisher for the arm controller
        self.arm_pub = rospy.Publisher(
            '/open_manipulator_6dof/arm_controller/command',
            JointTrajectory,
            queue_size=10
        )
        # Publisher for the gripper controller
        self.gripper_pub = rospy.Publisher(
            '/open_manipulator_6dof/gripper_controller/command',
            JointTrajectory,
            queue_size=10
        )
        self.link_attacher = GazeboLinkAttacher()
        rospy.loginfo("Gazebo publishers initialized.")
        rospy.sleep(1.0) # Wait for publishers to be ready

    def move_to_frame(self, target_frame, offset, execution_time=5.0, wait=True, rotation_lock=True):
        """
        Moves the robot arm to a target pose defined by a TF frame and an offset.

        :param target_frame: The name of the target TF frame (without leading '/').
        :param offset: A list [x, y, z] representing the offset from the target frame's origin.
        :param execution_time: The time (in seconds) the movement should take.
        :param wait: Boolean, whether to wait for the movement to complete.
        :param rotation_lock: Boolean, whether to lock the end-effector's rotation.
        """
        rospy.loginfo(f"Attempting to move to TF frame '{target_frame}' with offset {offset}.")

        try:
            # Wait for the transform to be available to avoid timing issues
            self.listener.waitForTransform(self.base_frame, target_frame, rospy.Time(0), rospy.Duration(4.0))

            # Get the transform (position and orientation) of the target frame relative to the base frame
            (trans, rot) = self.listener.lookupTransform(self.base_frame, target_frame, rospy.Time(0))

            # Calculate the final target pose by applying the offset to the frame's position
            # CORRECTED: Create a Pose message instead of a dictionary for the IK solver.
            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x = trans[0] + offset[0]
            target_pose.position.y = trans[1] + offset[1]
            target_pose.position.z = trans[2] + offset[2]
            


            rospy.loginfo(f"Target frame '{target_frame}' found at {trans}. Moving to final pose.")
            
            # Call the existing method to move the arm to the calculated pose
            self.move_arm_to_pose(
                target_pose,
                execution_time=execution_time,
                wait=wait,
                rotation_lock=rotation_lock
            )

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform for '{target_frame}': {e}")


    def move_arm_to_pose(self, target_pose, execution_time=5.0,wait=True, rotation_lock=True):
        """
        Moves the robot arm to a specified pose using inverse kinematics.

        :param target_pose: A dictionary with 'position' and 'orientation' keys.
                            'position' is a list [x, y, z] in meters.
                            'orientation' is a quaternion [x, y, z, w].
        :param execution_time: The time (in seconds) the movement should take.
        """
        rospy.loginfo(f"Moving arm to pose: {target_pose} in {execution_time}s.")
        joint_positions=None
        i=0

        while joint_positions is None and i < 40:
            rotation= quaternion_from_euler(0,np.pi-(i*np.pi/40) , 0)  # Default orientation (no rotation)

            target_pose.orientation.x = rotation[0]
            target_pose.orientation.y = rotation[1]
            target_pose.orientation.z = rotation[2]
            target_pose.orientation.w = rotation[3]

            # Get the joint positions using inverse kinematics
            joint_positions = get_ik_solution(target_pose,rotation_lock=rotation_lock)
            if len(joint_positions) != 6:
                rospy.logwarn(f"IK solution not found on attempt {i+1}. Retrying with different orientation.")
                joint_positions = None

            i += 1

        if joint_positions is None:
            rospy.logerr("Failed to compute IK solution.")
            return

        # Move the arm to the computed joint positions
        self._move_arm(joint_positions, execution_time=execution_time,wait=wait)
        
        if wait:
            rospy.sleep(execution_time)  # Wait for the movement to complete

    
    def _move_arm(self, joint_positions, execution_time=5.0,wait=True):
        """
        Moves the robot arm to a specified joint configuration.

        :param joint_positions: A list of 6 target joint angles (in radians).
        :param execution_time: The time (in seconds) the movement should take.
        """
        if len(joint_positions) != len(self.arm_joint_names):
            rospy.logerr(f"Invalid number of joint positions. Expected {len(self.arm_joint_names)}, got {len(joint_positions)}.")
            return

        rospy.loginfo(f"Moving arm to: {joint_positions} in {execution_time}s.")

        # Create the trajectory message
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.joint_names = self.arm_joint_names

        # Create a single trajectory point
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = rospy.Duration.from_sec(execution_time)

        # Add the point to the trajectory
        arm_msg.points.append(point)

        # Publish the message
        self.arm_pub.publish(arm_msg)
        rospy.loginfo("Arm command published. Waiting for completion...")
        if wait:
            rospy.sleep(execution_time)
        rospy.loginfo("Arm movement finished.")


    def move_gripper(self, grip=False, release=False, execution_time=2.0):
        """
        Controls the gripper to open or close.

        :param action: A string, either 'open' or 'close'.
        :param execution_time: The time (in seconds) the action should take.
        """
        # Create the trajectory message
        gripper_msg = JointTrajectory()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.joint_names = self.gripper_joint_names

        # Define joint positions based on the action
        point = JointTrajectoryPoint()
        if release:
            point.positions = [0.019, 0.019]  # Corresponds to a fully open gripper
        elif grip:
            point.positions = [-0.010, -0.010] # Corresponds to a closed gripper
        
        point.time_from_start = rospy.Duration.from_sec(execution_time)

        # Add the point to the trajectory
        gripper_msg.points.append(point)

        # Publish the message
        self.gripper_pub.publish(gripper_msg)
        rospy.loginfo("Gripper command published. Waiting for completion...")
        rospy.sleep(execution_time)
        rospy.loginfo("Gripper action finished.")

    def go_to_home(self, execution_time=5.0):
        """
        Moves the arm to a predefined home position.
        """
        rospy.loginfo("Moving arm to home position.")

        # Define the home joint positions (these should be set according to your robot's configuration)
        self._move_arm(self.home_position, execution_time)

    def grap_object(self, object_name, execution_time=2.0):
        """
        Grabs an object by attaching the gripper to it.

        :param object_name: The name of the object to grab.
        :param execution_time: The time (in seconds) the action should take.
        """
        rospy.loginfo(f"Grabbing object '{object_name}'...")

        self.move_to_frame(
            target_frame=object_name, 
            offset=[0.0, 0.0, 0.05],  # Adjust the offset as needed to position the gripper above the object
            execution_time=execution_time,
            wait=True,
            rotation_lock=True  
        )

        self.move_to_frame(
            target_frame=object_name, 
            offset=[0.0, 0.0, 0.015],  # Adjust the offset as needed to position the gripper above the object
            execution_time=execution_time,
            wait=True,
            rotation_lock=True  
        )

        self.move_gripper(grip=True, execution_time=execution_time)

        self.move_to_frame(
            target_frame=object_name, 
            offset=[0.0, 0.0, 0.05],  # Adjust the offset as needed to position the gripper above the object
            execution_time=execution_time,
            wait=False,
            rotation_lock=True  
        )
        rospy.sleep(2.0)  # Wait for the gripper to close before attaching
        # Attach the gripper link to the object link
        self.link_attacher.attach(
            model_name_1='robot',
            link_name_1='link7',
            model_name_2=object_name,
            link_name_2='link'
        )
        rospy.sleep(execution_time)  # Wait for the gripper to close
        self.go_to_home(execution_time=execution_time)  # Move back to home position after grabbing


    def release_object(self, object_name,release_object=None,z_offcet=0.035, execution_time=2.0):
        """
        Releases an object by detaching the gripper from it.

        :param object_name: The name of the object to release.
        :param execution_time: The time (in seconds) the action should take.
        """
        rospy.loginfo(f"Releasing object '{object_name}'...")

        self.move_to_frame(
            target_frame=object_name, 
            offset=[0.0, 0.0, z_offcet],  # Adjust the offset as needed to position the gripper above the object
            execution_time=execution_time,
            wait=True,
            rotation_lock=True  
        )


        # Detach the gripper link from the object link
        self.link_attacher.detach(
            model_name_1='robot',
            link_name_1='link7',
            model_name_2=release_object,
            link_name_2='link'
        )
        self.move_gripper(release=True, execution_time=execution_time)

        
        self.move_to_frame(
            target_frame=object_name, 
            offset=[0.0, 0.0, 0.07],  # Adjust the offset as needed to position the gripper above the object
            execution_time=execution_time,
            wait=True,
            rotation_lock=True  
        )
        rospy.loginfo(f"Object '{object_name}' released successfully.")
        self.go_to_home(execution_time=execution_time)  # Move back to home position after releasing 


    def Make_A_Move(self,Moving_Object, target_frame, offset, execution_time=5.0, wait=True, rotation_lock=True):
        """
        Moves the robot arm to a target pose defined by a TF frame and an offset.

        :param Moving_Object: The name of the object to move.
        :param target_frame: The name of the target TF frame (without leading '/').
        :param offset: A list [x, y, z] representing the offset from the target frame's origin.
        :param execution_time: The time (in seconds) the movement should take.
        :param wait: Boolean, whether to wait for the movement to complete.
        :param rotation_lock: Boolean, whether to lock the end-effector's rotation.
        """
        rospy.loginfo(f"Attempting to move {Moving_Object} to TF frame '{target_frame}' with offset {offset}.")
        self.grap_object(
            object_name=Moving_Object, 
            execution_time=execution_time
        )

        rospy.loginfo(f"Moving {Moving_Object} to frame '{target_frame}' with offset {offset}.")

        self.release_object(
            object_name=target_frame, 
            release_object=Moving_Object,
            z_offcet=0.05,  # Adjust the Z offset as needed to position the gripper above the objects
            execution_time=execution_time
        )
        rospy.loginfo(f"Successfully moved {Moving_Object} to frame '{target_frame}' with offset {offset}.")

if __name__ == "__main__":
    rospy.init_node("gazebo_controller_example", anonymous=True)
    
    # Initialize the controller
    controller = GazeboController()

    # --- Example 1: Move the arm to a hardcoded target pose ---
    controller.go_to_home(execution_time=5.0)  # Move to home position first
    # Open and close the gripper
    controller.move_gripper(grip=True, execution_time=2.0)
    controller.move_gripper(release=True, execution_time=2.0)

    # --- Example 2: Move the arm to a TF frame ---
    rospy.loginfo("--- Running Example 2: Move to TF frame ---")
    # NOTE: Replace 'unit_box' with the actual name of the TF frame of your object in Gazebo.
    # The object must be broadcasting its TF frame for this to work.
    try:
        cube_frame_name = 'red_cube'

        move_frame='blue_cube'  # The TF frame you want to move to
        # Define an offset to position the gripper above the cube, not inside it.
        # This moves the gripper 15cm on the Z-axis relative to the cube's origin.
        
        controller.Make_A_Move(
            Moving_Object=cube_frame_name,
            target_frame=move_frame, 
            offset=[0.0, 0.0, 0.05],  # Adjust the offset as needed to position the gripper above the object
            execution_time=5.0,
            wait=True,
            rotation_lock=True  
        )

    except Exception as e:
        rospy.logerr(f"Could not complete 'move_to_frame' example. Please ensure the TF frame exists in your simulation. Error: {e}")

