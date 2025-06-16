import rospy
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, DeleteModelResponse
import time

class CubeDeleter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('cube_deleter', anonymous=True)
        rospy.loginfo("Cube Deleter Node Initialized.")

        # Wait for the /gazebo/delete_model service to be available
        rospy.loginfo("Waiting for /gazebo/delete_model service...")
        rospy.wait_for_service('/gazebo/delete_model')
        self.delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        rospy.loginfo("Service /gazebo/delete_model is available.")

    def delete_cube(self, model_name):
        """
        Deletes a model from Gazebo.
        :param model_name: The name of the model to delete (e.g., 'red_cube').
        """
        rospy.loginfo(f"Attempting to delete model: {model_name}...")

        # Create the DeleteModel request
        req = DeleteModelRequest()
        req.model_name = model_name

        try:
            resp = self.delete_model_service(req)
            if resp.success:
                rospy.loginfo(f"Successfully deleted model: {model_name}.")
                return True
            else:
                rospy.logerr(f"Failed to delete model {model_name}: {resp.status_message}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for {model_name}: {e}")
            return False

if __name__ == '__main__':
    deleter = CubeDeleter()

    # Give Gazebo a moment to ensure models are spawned if this runs too quickly after spawning
    # Or, if running independently, ensure Gazebo is up and models exist.
    time.sleep(1) 

    # List of cube names to delete
    cubes_to_delete = ["red_cube", "green_cube", "blue_cube"]

    for cube_name in cubes_to_delete:
        deleter.delete_cube(cube_name)
        time.sleep(0.5) # Small delay between deletions

    rospy.loginfo("Finished attempting to delete all specified cubes.")
    rospy.spin() # Keep the node alive if you want to inspect logs