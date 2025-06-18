
import rospy
import actionlib
import pinocchio as pin
import numpy as np
from urdf_parser_py.urdf import URDF
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def get_pinocchio_model():
    """
    Loads the robot model from the URDF on the parameter server using Pinocchio.
    """
    rospy.loginfo("Loading robot model from parameter server...")
    # Load the URDF from the parameter server
    robot_urdf = URDF.from_parameter_server()
    # Convert it to a string
    robot_urdf_string = robot_urdf.to_xml_string()
    # Build the Pinocchio model from the URDF string
    model = pin.buildModelFromXML(robot_urdf_string)
    rospy.loginfo("Robot model loaded successfully.")
    return model

def solve_inverse_kinematics(model, target_pose):
    """
    Solves for the joint angles needed to reach a target pose using Pinocchio IK.
    """
    rospy.loginfo("Solving Inverse Kinematics...")
    data = model.createData()
    
    # The ID of the frame we want to control (the end-effector)
    # You can find frame names by inspecting the URDF or using `rosrun tf tf_echo`
    EE_FRAME_ID = model.getFrameId('end_effector_link')

    # Desired pose of the end-effector (a Pinocchio SE3 object)
    # pin.SE3(rotation_matrix, position_vector)
    oMdes = pin.SE3(target_pose['rotation'], np.array(target_pose['position']))

    # Initial guess for the joint configuration
    q0 = pin.neutral(model)

    # Damping factor for the IK solver
    eps = 1e-4
    # Max iterations
    IT_MAX = 1000
    # Damping factor
    damp = 1e-6
    
    q = q0.copy()
    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        # Get the current pose of the end-effector
        iMd = data.oMf[EE_FRAME_ID].inverse()
        # Calculate the error between desired and current pose
        err = pin.log(iMd.act(oMdes)).vector
        
        if np.linalg.norm(err) < eps:
            rospy.loginfo("IK converged successfully!")
            break
        
        if i >= IT_MAX:
            rospy.logwarn("IK did not converge after %d iterations." % IT_MAX)
            return None
        
        # Calculate the Jacobian for the end-effector frame
        J = pin.computeFrameJacobian(model, data, q, EE_FRAME_ID)
        # Compute the change in joint angles
        v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        # Update the joint configuration
        q = pin.integrate(model, q, v)
        i += 1

    # We only need the first 6 joint angles for the arm
    return q[:6]

def send_joint_goal(joint_angles):
    """
    Sends a joint trajectory goal with a single point to the arm controller.
    """
    if joint_angles is None:
        rospy.logerr("Cannot send goal, joint angles are invalid.")
        return

    client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for action server...")
    client.wait_for_server()
    rospy.loginfo("Action server found!")

    goal = FollowJointTrajectoryGoal()
    trajectory = JointTrajectory()
    trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    
    point = JointTrajectoryPoint()
    point.positions = joint_angles
    point.time_from_start = rospy.Duration(5.0) # Time to reach the target
    trajectory.points.append(point)
    goal.trajectory = trajectory

    rospy.loginfo("Sending goal with joint angles: %s" % str(joint_angles))
    client.send_goal(goal)
    client.wait_for_result()
    rospy.loginfo("Trajectory execution finished.")

if __name__ == '__main__':
    try:
        rospy.init_node('pinocchio_ik_control_node')
        
        # 1. Load the robot model using Pinocchio
        model = get_pinocchio_model()
        
        # 2. Define the target pose for the end-effector
        #    Position: [x, y, z] in meters
        #    Rotation: A 3x3 rotation matrix (here, facing down)
        target_pose = {
            'position': [0.15, 0.1, 0.25],
            'rotation': np.array([
                [0, 0, 1],
                [0, -1, 0],
                [1, 0, 0]
            ])
        }
        
        # 3. Solve for the required joint angles using Pinocchio's IK
        joint_solution = solve_inverse_kinematics(model, target_pose)
        
        # 4. Command the robot to move to the solved configuration
        send_joint_goal(joint_solution)

    except rospy.ROSInterruptException:
        print ("Program interrupted before completion.")
    except Exception as e:
        rospy.logerr("An error occurred: %s" % e)