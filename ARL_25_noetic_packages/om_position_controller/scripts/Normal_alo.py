#!/usr/bin/env python3

# Standard Library Imports
import json

# ROS-related Imports
import rospy
from std_msgs.msg import String, Char

# --- Constants and Configuration ---

# The number of disks in the puzzle.
N_DISKS = 3

# Defines the names of the pegs. This order is used for reference.
PEGS = ['A', 'B', 'C']


class HanoiStateBasedSolverNode:
    """
    A ROS node that uses a robust recursive algorithm to determine the next
    optimal move in the Tower of Hanoi puzzle from any valid game state.
    """
    def __init__(self):
        """Initializes the ROS node, subscribers, and publishers."""
        rospy.init_node('hanoi_state_based_solver_node', anonymous=True)

        self.last_game_state_json = None
        rospy.Subscriber('/hanoi/game_state', String, self.game_state_callback)
        rospy.Subscriber('/hanoi/ask_llm', String, self.trigger_move_callback)
        self.move_publisher = rospy.Publisher('/hanoi/move', String, queue_size=10)
        self.reset_publisher = rospy.Publisher('/hanoi_reset', Char, queue_size=10)
        
        rospy.loginfo("Truly Robust State-Based Hanoi Solver Node initialized.")
        rospy.loginfo(f"Solving for {N_DISKS} disks from any valid state.")
        rospy.loginfo("Subscribing to /hanoi/game_state and /hanoi/ask_llm.")

    def game_state_callback(self, msg):
        """Stores the latest state of the game upon receiving it."""
        self.last_game_state_json = msg.data

    def trigger_move_callback(self, msg):
        """
        Triggered by a message, this function analyzes the current game state,
        calculates the single best next move, and publishes it.
        """
        rospy.loginfo("--- Received Signal to Calculate Next Move ---")
        
        if self.last_game_state_json is None:
            rospy.logwarn("Trigger received, but no game state has been received yet. Ignoring.")
            return

        try:
            state_dict = json.loads(self.last_game_state_json)
        except json.JSONDecodeError:
            rospy.logerr(f"Failed to decode JSON state: {self.last_game_state_json}")
            return

        # --- 1. State Validation and Puzzle Completion Check ---
        is_valid, error_message = self.validate_state(state_dict)
        if not is_valid:
            rospy.logwarn(f"Invalid State Detected: {error_message}")
            self.reset_publisher.publish(Char(data=ord('1')))
            return

        if len(state_dict.get('C', [])) == N_DISKS:
            rospy.loginfo(f"Puzzle Solved! All {N_DISKS} disks are on Peg C.")
            return

        # --- 2. Calculate Move using a Truly Robust Algorithm ---
        rospy.loginfo("Calculating best move using robust stateless algorithm...")
        # The main goal is to get a tower of size N_DISKS to peg 'C'.
        move = self.calculate_next_move(N_DISKS, 'C', state_dict)

        if move is None:
            rospy.logwarn("Move calculation resulted in no action. This may happen if the puzzle is already solved.")
            return
        
        source_peg, dest_peg = move
        rospy.loginfo(f"State-based calculation: Best move is from {source_peg} to {dest_peg}")
        
        # --- 3. Publish Move ---
        move_command = {"source_peg": source_peg, "destination_peg": dest_peg}
        move_json_string = json.dumps(move_command)

        rospy.sleep(0.1)
        self.move_publisher.publish(move_json_string)
        rospy.loginfo(f"Published move to /hanoi/move: {move_json_string}")
        rospy.loginfo("-------------------------------------------")

    def calculate_next_move(self, h, target_peg, state):
        """
        Calculates the next move required to get a tower of height 'h' to the target_peg.
        This algorithm is stateless and robust against non-optimal board states.

        Args:
            h (int): The height of the tower to move.
            target_peg (str): The peg where the tower should end up.
            state (dict): The current state of all disks.

        Returns:
            tuple or None: A (source, destination) move, or None if the tower is already in place.
        """
        if h <= 0:
            return None

        # Find the current location of the largest disk in this sub-problem.
        peg_of_h = [peg for peg, disks in state.items() if h in disks][0]

        # If the largest disk 'h' is already at the target...
        if peg_of_h == target_peg:
            # ...then the problem is reduced to moving the smaller tower (h-1) to that same target.
            return self.calculate_next_move(h - 1, target_peg, state)
        
        # If the largest disk 'h' is NOT at the target...
        else:
            # ...our goal is to move it there. To do so, we must first move the
            # h-1 tower to the third "auxiliary" peg to get it out of the way.
            aux_peg = [p for p in PEGS if p != peg_of_h and p != target_peg][0]
            
            # We recursively find the next move for this new, smaller sub-problem.
            move_for_smaller_tower = self.calculate_next_move(h - 1, aux_peg, state)
            
            # If the recursive call returns a move, that's our action.
            if move_for_smaller_tower:
                return move_for_smaller_tower
            # If it returns None, it means the smaller tower is already in place.
            # Therefore, the correct action now is to move disk 'h'.
            else:
                return (peg_of_h, target_peg)

    def validate_state(self, state_dict):
        """Checks if the current state is valid according to Hanoi rules."""
        for peg, disks in state_dict.items():
            for i in range(len(disks) - 1):
                if disks[i] < disks[i+1]:
                    error = f"Invalid state on Peg {peg}: disk {disks[i+1]} cannot be on disk {disks[i]}."
                    return False, error
        return True, ""


if __name__ == '__main__':
    try:
        HanoiStateBasedSolverNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred: {e}")