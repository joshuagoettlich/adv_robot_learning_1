#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse, DeleteModel, DeleteModelRequest
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Char
import os
import rospkg
import tf
import random
import time
from delete_cubes import CubeDeleter

class CubeSpawner:
    def __init__(self):
        """
        Initializes the CubeSpawner node.
        - Sets up ROS node and services for spawning/deleting models.
        - Defines peg locations and cube properties.
        - Initializes TF broadcaster and ROS publishers for peg data.
        - Subscribes to a control topic to reset the scenario.
        """
        rospy.init_node('cube_spawner', anonymous=True)
        rospy.loginfo("Cube Spawner Node Initialized.")
        deleter = CubeDeleter()

        # Give Gazebo a moment
        time.sleep(1)

        # List of cube names to delete
        cubes_to_delete = ["red_cube", "green_cube", "blue_cube"]

        for cube_name in cubes_to_delete:
            deleter.delete_cube(cube_name)
            time.sleep(0.5) # Small delay between deletions

        # --- Define Peg Locations and Cube Properties ---
        self.peg_names = ['A', 'B', 'C']
        self.peg_locations = [
            Point(0.13, -0.1, 0),  # Peg A
            Point(0.13, 0.0, 0),    # Peg B
            Point(0.13, 0.1, 0)    # Peg C
        ]

        self.cubes = {
            'blue_cube':  {'name': 'blue_cube',  'size': 2, 'height': 0.02},
            'green_cube': {'name': 'green_cube', 'size': 3, 'height': 0.02},
            'red_cube':   {'name': 'red_cube',   'size': 1, 'height': 0.02}
        }

        # --- ROS Setup ---
        rospack = rospkg.RosPack()
        try:
            self.pkg_path = rospack.get_path('om_position_controller')
            rospy.loginfo(f"Found 'om_position_controller' package at: {self.pkg_path}")
        except rospkg.ResourceNotFound:
            rospy.logerr("Package 'om_position_controller' not found. Please ensure it's in your ROS workspace.")
            self.pkg_path = None

        # --- Gazebo Service Proxies ---
        rospy.loginfo("Waiting for Gazebo services...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        rospy.wait_for_service('/gazebo/delete_model')
        self.delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        rospy.loginfo("Gazebo services are available.")

        # --- Publishers and TF Broadcaster ---
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.peg_publishers = {}
        for name in self.peg_names:
            topic_name = f'/peg_{name}/pose'
            self.peg_publishers[name] = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
            rospy.loginfo(f"Publishing peg pose on topic: {topic_name}")

        # --- Subscriber ---
        rospy.Subscriber('/hanoi_reset', Char, self.reset_scenario_callback)
        rospy.loginfo("Subscribed to /hanoi_reset topic for reset commands.")

    def _read_sdf_file(self, model_name):
        """Reads the content of an SDF file."""
        if not self.pkg_path: return None
        file_path = os.path.join(self.pkg_path, 'models', f'{model_name}.sdf')
        if not os.path.exists(file_path):
            rospy.logerr(f"SDF file not found: {file_path}")
            return None
        with open(file_path, 'r') as f:
            return f.read()

    def spawn_cube(self, model_name, pose):
        """Spawns a cube model in Gazebo at a given pose."""
        sdf_xml = self._read_sdf_file(model_name)
        if not sdf_xml: return False
        req = SpawnModelRequest(model_name=model_name, model_xml=sdf_xml, initial_pose=pose, reference_frame="world")
        try:
            resp = self.spawn_model_service(req)
            if resp.success:
                rospy.loginfo(f"Successfully spawned {model_name}.")
                return True
            else:
                rospy.logerr(f"Failed to spawn {model_name}: {resp.status_message}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for {model_name}: {e}")
            return False

    def delete_all_cubes(self):
        """Deletes all defined cubes from the Gazebo simulation."""
        rospy.loginfo("Deleting all cubes...")
        for model_name in self.cubes.keys():
            req = DeleteModelRequest(model_name=model_name)
            try:
                resp = self.delete_model_service(req)
                if resp.success:
                    rospy.loginfo(f"Successfully deleted {model_name}.")
                else:
                    rospy.logwarn(f"Failed to delete {model_name}: {resp.status_message}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed for {model_name}: {e}")
            time.sleep(0.1)

    def generate_and_spawn_hanoi_scenario(self):
        """Generates a random, valid Tower of Hanoi configuration and spawns the cubes."""
        rospy.loginfo("Generating random Tower of Hanoi scenario...")
        pegs = [[], [], []]
        cubes_to_place = list(self.cubes.values())
        random.shuffle(cubes_to_place)

        for cube_info in cubes_to_place:
            pegs[random.randint(0, 2)].append(cube_info)

        for peg in pegs:
            peg.sort(key=lambda c: c['size'], reverse=True)

        for i, peg in enumerate(pegs):
            peg_base_pos = self.peg_locations[i]
            current_z = 0.02
            if not peg: continue

            peg_name = self.peg_names[i]
            rospy.loginfo(f"Spawning for Peg {peg_name}: {[c['name'] for c in peg]}")
            for cube in peg:
                pose = Pose(position=Point(peg_base_pos.x, peg_base_pos.y, current_z), orientation=Quaternion(0,0,0,1))
                self.spawn_cube(cube['name'], pose)
                rospy.sleep(0.5)
                current_z += cube['height']

    def reset_scenario_callback(self, msg):
        """Callback to delete all cubes and spawn a new scenario."""
   
        rospy.loginfo("Reset command '1' received on /hanoi_reset.")
        self.delete_all_cubes()
        time.sleep(1.0)
        self.generate_and_spawn_hanoi_scenario()
       
    def publish_peg_data(self, event=None):
        """Publishes the static locations of the three pegs to TF and as ROS topics."""
        now = rospy.Time.now()
        for i, peg_pos in enumerate(self.peg_locations):
            peg_name = self.peg_names[i]
            
            # 1. Publish TF Transform
            self.tf_broadcaster.sendTransform(
                (peg_pos.x, peg_pos.y, peg_pos.z),
                (0, 0, 0, 1),
                now,
                f"peg_{peg_name}",
                "world"
            )

            # 2. Publish ROS Topic
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position = peg_pos
            pose_msg.pose.orientation = Quaternion(0, 0, 0, 1)
            
            if peg_name in self.peg_publishers:
                self.peg_publishers[peg_name].publish(pose_msg)

if __name__ == '__main__':
    try:
        spawner = CubeSpawner()
        rospy.sleep(2)

        # Initial spawn on startup
        spawner.generate_and_spawn_hanoi_scenario()

        # Timer to continuously publish peg data (TF and topics)
        rospy.Timer(rospy.Duration(0.1), spawner.publish_peg_data)

        rospy.loginfo("Scenario ready. Listening on /hanoi_reset for reset command ('1').")
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Cube Spawner node shut down.")