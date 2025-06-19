#!/usr/bin/env python3

import rospy
import json
from std_msgs.msg import String
from Movement_wrapper import GazeboController, get_cube_position
from std_msgs.msg import Char


# --- Configuration ---
# This maps the disk numbers from the game logic (e.g., 1, 2, 3)
# to the TF frame names used in Gazebo/RViz.
# !! IMPORTANT !!: Make sure these names match the model names in your Gazebo world.
DISK_TF_MAP = {
    1: "red_cube",
    2: "blue_cube",
    3: "green_cube",
}

# This maps the peg letters from the game logic (A, B, C)
# to the TF frame names of the pegs in Gazebo.
PEG_TF_MAP = {
    'A': "peg_A",
    'B': "peg_B",
    'C': "peg_C",
}

# The total number of disks in the puzzle. Used to check for the win condition.
TOTAL_DISKS = 3

class HanoiMainController:
    """
    Orchestrates the Tower of Hanoi puzzle by interfacing between the LLM
    logic node and the robot's movement controller.
    """
    def __init__(self):
        """
        Initializes the main controller node.
        """
        rospy.init_node("hanoi_main_controller", anonymous=True)

        # Publisher to trigger the LLM to think about the next move.
        self.ask_llm_pub = rospy.Publisher('/hanoi/ask_llm', String, queue_size=10)

        # Initialize the robot controller from your wrapper script.
        self.robot_controller = GazeboController()

        rospy.loginfo("Hanoi Main Controller is initialized and ready.")
        rospy.sleep(1) # Give publishers a moment to connect.

    def ask_for_next_move(self):
        """
        Publishes a message to the /hanoi/ask_llm topic to signal the
        LLM solver node to determine the next move.
        """
        rospy.loginfo("Asking LLM for the next move...")
        # The content of the message doesn't matter, it's just a signal.
        self.ask_llm_pub.publish(String(data="go"))

    def wait_for_move_command(self):
        """
        Waits for a single message on the /hanoi/move topic and returns
        the parsed JSON data.
        """
        rospy.loginfo("Waiting for a move command from the LLM on /hanoi/move...")
        try:
            # This is a blocking call that waits for one message.
            move_msg = rospy.wait_for_message('/hanoi/move', String, timeout=40.0)
            rospy.loginfo(f"Received move command: {move_msg.data}")
            return json.loads(move_msg.data)
        except rospy.ROSException:
            rospy.logerr("Timed out waiting for a move command. Is the LLM node running?")
            return None
        except json.JSONDecodeError:
            rospy.logerr(f"Failed to decode JSON from move command: {move_msg.data}")
            return None

    def get_latest_game_state(self):
        """
        Retrieves the most recent game state from the /hanoi/game_state topic.
        """
        rospy.loginfo("Fetching the latest game state from /hanoi/game_state...")
        try:
            # This is a blocking call that waits for one message.
            state_msg = rospy.wait_for_message('/hanoi/game_state', String, timeout=10.0)
            rospy.loginfo(f"Received game state: {state_msg.data}")
            return json.loads(state_msg.data)
        except rospy.ROSException:
            rospy.logerr("Timed out waiting for game state. Is the state publisher running?")
            return None
        except json.JSONDecodeError:
            rospy.logerr(f"Failed to decode JSON from game state: {state_msg.data}")
            return None

    def run_game_loop(self):
        """
        The main loop that controls the entire puzzle-solving process.
        """
        # Start with the robot in a known safe position.
        self.robot_controller.go_to_home(execution_time=5.0)

        while not rospy.is_shutdown():
            # 1. Get the current game state to check for a win before acting.
            current_state = self.get_latest_game_state()
            if not current_state:
                rospy.logerr("Could not get game state. Stopping.")
                break

            # 2. Check for the win condition.
            if len(current_state.get('C', [])) == TOTAL_DISKS:
                rospy.loginfo("PUZZLE SOLVED! All disks are on Peg C.")
                break

            # 3. Ask the LLM for the next logical move.
            self.ask_for_next_move()

            # 4. Wait for the LLM to respond with a move.
            move_command = self.wait_for_move_command()
            if not move_command:
                rospy.logerr("Could not get move command. Stopping.")
                break

            # 5. Get the game state *again* to ensure we're acting on the most
            #    up-to-date information right before the move.
            current_state = self.get_latest_game_state()
            if not current_state:
                rospy.logerr("Could not get game state before executing move. Stopping.")
                break

            # 6. Determine the TF frames for the source and target.
            source_peg_id = move_command['source_peg']
            dest_peg_id = move_command['destination_peg']

            # The object to move is the top disk on the source peg.
            if not current_state[source_peg_id]:
                rospy.logerr(f"LLM requested move from empty peg {source_peg_id}. Skipping.")
                continue
            
            moving_disk_number = current_state[source_peg_id][-1]
            moving_object_tf = DISK_TF_MAP[moving_disk_number]

            # The target frame depends on if the destination peg has disks on it.
            destination_peg_disks = current_state.get(dest_peg_id, [])
            if not destination_peg_disks:
                # If the destination peg is empty, the target is the peg itself.
                target_frame_tf = PEG_TF_MAP[dest_peg_id]
            else:
                # If not empty, the target is the top disk on that peg.
                top_disk_number = destination_peg_disks[-1]
                target_frame_tf = DISK_TF_MAP[top_disk_number]
            
            rospy.loginfo(f"Executing move: {moving_object_tf} -> {target_frame_tf}")

            # 7. Execute the move using the Make_A_Move wrapper function.
            # NOTE: The Make_A_Move function in your wrapper has a hardcoded Z offset
            # for placement. We rely on that to stack the disks correctly.
            self.robot_controller.Make_A_Move(
                Moving_Object=moving_object_tf,
                target_frame=target_frame_tf,
                offset=[0.0, 0.0, 0.05], # Offset is not used by your Make_A_Move, but included for clarity.
                execution_time=8.0,
                wait=True
            )

            rospy.loginfo("Move execution complete. Waiting a moment for state to settle...")
            rospy.sleep(3.0) # Give time for Gazebo/TF to update after the move.
        
        rospy.loginfo("Hanoi Main Controller loop has finished.")
        # Return to home position at the end.
        self.robot_controller.go_to_home(execution_time=5.0)

if __name__ == '__main__':
    try:
        controller_node = HanoiMainController()
        controller_node.run_game_loop()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")

