#!/usr/bin/env python

import rospy
import tf
import math
from pprint import pprint

class HanoiGameState:
    """
    Monitors the state of the Tower of Hanoi game by listening to TF transforms.
    It determines which cubes are on which pegs and represents them by their
    size number (1, 2, 3).
    """
    def __init__(self):
        """
        Initializes the HanoiGameState node.
        - Sets up ROS node and a TF listener.
        - Defines the names of the game objects (pegs, cubes).
        - Creates a mapping from cube names to their integer sizes.
        """
        rospy.init_node('hanoi_state_monitor', anonymous=True)
        rospy.loginfo("Hanoi Game State Monitor Node Initialized.")

        self.tf_listener = tf.TransformListener()

        # --- Configuration ---
        self.peg_names = ['peg_A', 'peg_B', 'peg_C']
        self.cube_names = ['red_cube', 'green_cube', 'blue_cube']
        self.world_frame = 'world'
        self.xy_search_radius = 0.05  # 5 cm

        # --- MAPPING: From Cube Name to Number (Size) ---
        # This map translates the TF frame name of a cube to its logical size.
        # This information is based on the 'size' property in your spawner script.
        self.cube_size_map = {
            'red_cube': 1,
            'blue_cube': 2,
            'green_cube': 3
        }
        rospy.loginfo(f"Using cube name-to-size mapping: {self.cube_size_map}")

        rospy.loginfo("Waiting for TF transforms to become available...")
        rospy.sleep(2.0)
        rospy.loginfo("Ready to monitor game state.")

    def get_pose(self, target_frame):
        """
        Looks up the latest pose (translation) of a given frame relative to the world.
        """
        try:
            self.tf_listener.waitForTransform(self.world_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, target_frame, rospy.Time(0))
            return trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get pose for '{target_frame}': {e}")
            return None

    def determine_game_state(self):
        """
        Determines the current arrangement of cubes on the pegs.
        
        Returns:
            dict: A dictionary where keys are peg names ('A', 'B', 'C') and values
                  are lists of cube sizes (numbers), sorted from bottom to top.
                  Example: {'A': [3, 1], 'B': [], 'C': [2]}
        """
        # 1. Get current positions
        peg_poses = {name: self.get_pose(name) for name in self.peg_names}
        cube_poses = {name: self.get_pose(name) for name in self.cube_names}

        peg_poses = {k: v for k, v in peg_poses.items() if v is not None}
        cube_poses = {k: v for k, v in cube_poses.items() if v is not None}

        # 2. Associate cubes with pegs
        peg_contents = {name.split('_')[1]: [] for name in self.peg_names}

        for cube_name, cube_pos in cube_poses.items():
            for peg_name, peg_pos in peg_poses.items():
                dist_xy = math.sqrt((cube_pos[0] - peg_pos[0])**2 + (cube_pos[1] - peg_pos[1])**2)
                if dist_xy < self.xy_search_radius:
                    short_peg_name = peg_name.split('_')[1]
                    peg_contents[short_peg_name].append({'name': cube_name, 'z': cube_pos[2]})
                    break

        # 3. Sort by height and MAP TO NUMBER
        final_state = {}
        for peg_name, cubes_on_peg in peg_contents.items():
            # Sort cubes from lowest to highest z-coordinate
            sorted_cubes = sorted(cubes_on_peg, key=lambda c: c['z'])
            
            # *** KEY CHANGE IS HERE ***
            # Instead of using the cube name, look up its size number in our map.
            # Using .get() is safer than direct access; it returns None if a name is not found.
            final_state[peg_name] = [
                self.cube_size_map.get(cube['name']) for cube in sorted_cubes
            ]

        return final_state

    def run(self):
        """Main loop to continuously check and print the game state."""
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            current_state = self.determine_game_state()
            rospy.loginfo("--- Current Tower of Hanoi State (by Size) ---")
            pprint(current_state)
            rospy.loginfo("----------------------------------------------")
            rate.sleep()

if __name__ == '__main__':
    try:
        state_monitor = HanoiGameState()
        state_monitor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Hanoi State Monitor node shut down.")