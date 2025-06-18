#!/usr/bin/env python

import rospy
import sys
import geometry_msgs.msg
from moveit_msgs.msg import PositionIKRequest, RobotState, OrientationConstraint, JointConstraint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState

import math

def get_ik_solution(target_pose, rotation_lock=True):
    """
    Solves for Inverse Kinematics.

    Args:
        target_pose (geometry_msgs.msg.Pose): The desired pose of the end-effector.
        rotation_lock (bool): If False, the end-effector orientation is disregarded.
                              If True, MoveIt will try to match the orientation.

    Returns:
        list: A list of joint values for the solution, or an empty list if not found.
    """
    # 1. Wait for the /compute_ik service to become available
    rospy.loginfo("Waiting for '/compute_ik' service...")
    rospy.wait_for_service('/compute_ik')
    rospy.loginfo("Service is available.")

    try:
        # 2. Create a handle for the service
        ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        # 3. Build the service request
        request = GetPositionIKRequest()
        request.ik_request.group_name = "arm"  # Make sure this is your arm's planning group
        request.ik_request.ik_link_name = "end_effector_link" # Make sure this is your end-effector link
        
        # The service wants a RobotState as a seed, we can use an empty one
        request.ik_request.robot_state = RobotState()
        # NOTE: You should populate this with the actual joint names of your robot
        request.ik_request.robot_state.joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        request.ik_request.robot_state.joint_state.position = [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194] # Zeros are a common seed

        # Set the pose for the end-effector
        request.ik_request.pose_stamped.header.frame_id = "world" # Or your robot's base frame
        request.ik_request.pose_stamped.pose = target_pose

        joint_constraint = JointConstraint()

        joint_constraint.joint_name = 'joint1' # Make sure this is the correct joint name

        # Define the desired position and tolerances.
        # This will constrain the joint to be between 0.5 and 1.0 radians.
        # Desired middle position: 0.75
        # Lower bound: 0.75 - 0.25 = 0.5
        # Upper bound: 0.75 + 0.25 = 1.0
        joint_constraint.position = 0 
        joint_constraint.tolerance_above = 1
        joint_constraint.tolerance_below = 1
        
        # Set the weight for this constraint. A weight of 1.0 makes it a hard constraint.
        joint_constraint.weight = 0.5

        # Add the constraint to the IK request
        request.ik_request.constraints.joint_constraints.append(joint_constraint)

        
        # If rotation_lock is False, add a very loose orientation constraint
        if not rotation_lock:
            rospy.loginfo("Disregarding orientation (rotation_lock=False). Applying loose constraints.")
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = request.ik_request.pose_stamped.header.frame_id
            orientation_constraint.link_name = request.ik_request.ik_link_name
            # A null orientation means no specific orientation is desired.
            orientation_constraint.orientation.w = 0.0 
            # Set very large tolerances
            orientation_constraint.absolute_x_axis_tolerance = math.pi
            orientation_constraint.absolute_y_axis_tolerance = math.pi
            orientation_constraint.absolute_z_axis_tolerance = math.pi
    
            orientation_constraint.weight = 0.1
            request.ik_request.constraints.orientation_constraints.append(orientation_constraint)
        else:
            rospy.loginfo("Attempting to match orientation (rotation_lock=True).")

        # Other parameters
        request.ik_request.timeout = rospy.Duration(1.0)
        request.ik_request.avoid_collisions = True

        # 4. Call the service
        rospy.loginfo("Sending IK request...")
        response = ik_service(request)

        # 5. Check the response and print the result
        if response.error_code.val == response.error_code.SUCCESS:
            rospy.loginfo("IK Solution Found!")
            rospy.loginfo("IK Solution (Joint Names): %s", response.solution.joint_state.name)
            rospy.loginfo("IK Solution (Joint Values): %s", response.solution.joint_state.position)
            return response.solution.joint_state.position[:6]  # Return only the first 6 joint values
        else:
            rospy.logerr("IK Failed with error code: %d", response.error_code.val)
            # You can check moveit_msgs/MoveItErrorCodes.msg for the meaning of other codes
            return []

    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return []


if __name__ == '__main__':
    try:
        rospy.init_node('service_based_ik_solver_example')

        # Define the target pose you want to solve for
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0 # Neutral orientation (facing "forward")
        pose_goal.position.x = 0.3
        pose_goal.position.y = 0.1
        pose_goal.position.z = 0.4
        
        print("\n" + "="*30)
        print(" Case 1: Solving IK while disregarding orientation ")
        print("="*30)
        solution_unlocked = get_ik_solution(pose_goal, rotation_lock=False)
        if solution_unlocked:
            print("Found Solution (rotation unlocked): {}".format(solution_unlocked))
        else:
            print("No solution found.")

        print("\n" + "="*30)
        print(" Case 2: Solving IK while enforcing orientation ")
        print("="*30)
        # For some robots, the exact orientation might be impossible to reach
        # along with the position, so this might fail more often.
        solution_locked = get_ik_solution(pose_goal, rotation_lock=True)
        if solution_locked:
            print("Found Solution (rotation locked): {}".format(solution_locked))
        else:
            print("No solution found.")


    except rospy.ROSInterruptException:
        pass