#!/root/catkin_ws/gemini_env/bin/python3

import rospy
import json
import requests

# Import the standard String message type.
from std_msgs.msg import String
from std_msgs.msg import Char


SYSTEM_PROMPT = """
You are an AI that solves the 3-disk Tower of Hanoi puzzle. Your goal is to move all disks to Peg C in the order [3, 2, 1].

Given the current arrangement of disks on Pegs A, B, and C, tell me the single best next move.

Rules:

    Move only one disk at a time.
    Only the top disk of a peg can be moved.
    Never place a larger disk onto a smaller one. (e.g., Disk 3 cannot be on Disk 2).

Instructions:

    First, check if the provided Current State is valid according to the rules.
    If the state is invalid, respond with: {"error": "Invalid state."}
    If the state is valid, respond with the single best move in JSON format. Example: {"source_peg": "A", "destination_peg": "C"}

Provide only the JSON response and no other text.
Examples for Context

Disks are numbered 1 (smallest), 2 (medium), and 3 (largest). The lists represent pegs from bottom to top, so [3, 2, 1] means disk 1 is on top of 2, which is on top of 3.
5 Valid States

A state is valid if no larger disk is on a smaller disk.

    Peg A: [3, 2, 1], Peg B: [], Peg C: [] (The starting state is valid).
    Peg A: [3, 2], Peg B: [], Peg C: [1] (Disk 1 was moved to an empty peg).
    Peg A: [3], Peg B: [2], Peg C: [1] (Each peg is either empty or holds a valid stack).
    Peg A: [3], Peg B: [2, 1], Peg C: [] (Disk 1 was correctly placed on disk 2).
    Peg A: [], Peg B: [], Peg C: [3, 2, 1] (The goal state is valid).

5 Invalid States

A state is invalid if any peg has a larger disk on a smaller one.

    Peg A: [3, 1, 2], Peg B: [], Peg C: [] (Invalid: Disk 2 cannot be on Disk 1).
    Peg A: [1, 3], Peg B: [2], Peg C: [] (Invalid: Disk 3 cannot be on Disk 1).
    Peg A: [], Peg B: [2, 3, 1], Peg C: [] (Invalid: Disk 3 cannot be on Disk 2).
    Peg A: [2], Peg B: [1], Peg C: [3] (Invalid: This configuration is impossible to reach legally, but if given as input, it's a valid state. Let's make a better example. Peg A: [3], Peg B: [1, 2], Peg C: [] -> Invalid: Disk 2 cannot be on Disk 1).
    Peg A: [1], Peg B: [2], Peg C: [3, 2] (Invalid: This is physically impossible with one set of disks. A better example: Peg A: [2, 1], Peg B: [3], Peg C: [] -> Invalid: Disk 1 can be on 2, but let's assume the error is elsewhere: Peg A: [3], Peg B: [], Peg C: [1, 2] -> Invalid: Disk 2 cannot be on Disk 1).

5 Valid Moves

A move is valid if it takes the top disk and places it on an empty peg or a larger disk.

    State: Peg A: [3, 2, 1], Peg B: [], Peg C: [] Move: A -> C (Moves disk 1 to empty Peg C).
    State: Peg A: [3, 2], Peg B: [], Peg C: [1] Move: A -> B (Moves disk 2 to empty Peg B).
    State: Peg A: [3], Peg B: [2], Peg C: [1] Move: C -> B (Moves disk 1 onto disk 2).
    State: Peg A: [], Peg B: [2, 1], Peg C: [3] Move: B -> A (Moves disk 1 to empty Peg A).
    State: Peg A: [1], Peg B: [2], Peg C: [3] Move: A -> C (Moves disk 1 onto disk 3).

5 Invalid Moves

A move is invalid if it breaks one of the rules.

    State: Peg A: [3, 2], Peg B: [1], Peg C: [] Move: A -> B (Invalid: Cannot place disk 2 on disk 1).
    State: Peg A: [3, 2, 1], Peg B: [], Peg C: [] Move: A -> C (moving disk 2) (Invalid: Can only move the top disk, which is disk 1).
    State: Peg A: [3], Peg B: [2], Peg C: [1] Move: A -> B (Invalid: Cannot place disk 3 on disk 2).
    State: Peg A: [3], Peg B: [1], Peg C: [2] Move: C -> B (Invalid: Cannot place disk 2 on disk 1).
    State: Peg A: [3, 2, 1], Peg B: [], Peg C: [] Move: B -> A (Invalid: Cannot move a disk from an empty peg).

"""

class GeminiHanoiSolverNode:
    def __init__(self):
        """,
        Init,ializes the ROS node, publishers, and subscribers.
        """
        rospy.init_node('gemini_hanoi_solver_node', anonymous=True)

        self.api_key = "REMOVED" # you can put your own API key here.
        if "YOUR_GOOGLE" in self.api_key:
            rospy.logerr("API KEY NOT SET in the script. Please edit the file and add your key.")
            rospy.signal_shutdown("API Key not found.")
            return

        # This variable will store the most recent game state received.
        self.last_game_state_json = None

        # Subscriber to continuously update the game state.
        rospy.Subscriber('/hanoi/game_state', String, self.game_state_callback)
        
        # Subscriber that acts as a signal to trigger the LLM query.
        rospy.Subscriber('/hanoi/ask_llm', String, self.trigger_llm_callback)
        
        # Publisher to send the next move command.
        self.move_publisher = rospy.Publisher('/hanoi/move', String, queue_size=10)

        self.reset_publisher = rospy.Publisher('/hanoi_reset', Char, queue_size=10)
        
        rospy.loginfo("Gemini Hanoi Solver Node initialized.")
        rospy.loginfo("Subscribing to /hanoi/game_state for state updates.")
        rospy.loginfo("Subscribing to /hanoi/ask_llm to trigger moves.")
        rospy.loginfo("Publishing moves to /hanoi/move.")

    def game_state_callback(self, msg):
        """
        This function is called whenever a new message is published on /hanoi/game_state.
        It stores the latest state of the game.
        """
        rospy.loginfo(f"Received new game state from /hanoi/game_state.")
        self.last_game_state_json = msg.data

    def trigger_llm_callback(self, msg):
        """
        This function is called when a signal is received on /hanoi/ask_llm.
        It uses the last known game state to query the LLM and publish a move.
        The content of the message `msg` is ignored; it's used only as a trigger.
        """
        rospy.loginfo("--- Received Signal on /hanoi/ask_llm ---")
        
        if self.last_game_state_json is None:
            rospy.logwarn("LLM triggered, but no game state has been received yet. Ignoring.")
            return

        try:
            state_dict = json.loads(self.last_game_state_json)
        except json.JSONDecodeError:
            rospy.logerr(f"Failed to decode the last stored JSON state: {self.last_game_state_json}")
            return

        # Extract the state of each peg.
        peg_a = list(state_dict.get('A', []))
        peg_b = list(state_dict.get('B', []))
        peg_c = list(state_dict.get('C', []))
        
        rospy.loginfo(f"Using last known state: Peg A: {peg_a}, Peg B: {peg_b}, Peg C: {peg_c}")
        rospy.loginfo("-------------------------------------------")

        # Check if the puzzle is solved.
        if len(peg_c) == 3: # Assuming 3 disks
            rospy.loginfo("Puzzle Solved! No further moves needed.")
            return

        # Format the current state for the LLM prompt.
        current_state = f"Peg A Disks: {peg_a}\n Peg B Disks: {peg_b}\n Peg C Disks: {peg_c}"
        full_prompt = f"{SYSTEM_PROMPT}\n\nCurrent State:\n{current_state}"
        
        rospy.loginfo("Querying LLM for the next move...")
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": full_prompt}]}]}
        while not rospy.is_shutdown():
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=20)
                response.raise_for_status()

                result_json = response.json()
                llm_text_response = result_json['candidates'][0]['content']['parts'][0]['text']
                
                cleaned_response = llm_text_response.strip().replace('```json', '').replace('```', '').strip()
                move_data = json.loads(cleaned_response)

                if 'error' in move_data:
                    error_message = move_data['error']
                    rospy.logwarn(f"LLM reported an invalid state: {error_message}")
                    rospy.loginfo("Publishing to /hanoi_reset due to invalid state.")
                    # Publish the error message to the reset topic.
                    self.reset_publisher.publish(String(data="1"))  # Sending '1' to trigger a reset.
                    break # Exit the loop after detecting and reporting the error.


                move_command = {
                    "source_peg": move_data['source_peg'],
                    "destination_peg": move_data['destination_peg']
                }
                move_json_string = json.dumps(move_command)

                self.move_publisher.publish(move_json_string)
                rospy.loginfo(f"Published move to /hanoi/move: {move_json_string}")
                break



            except requests.exceptions.RequestException as e:
                rospy.logerr(f"Network error calling API: {e}")
                break
            except (KeyError, IndexError) as e:
                rospy.logerr(f"Failed to parse LLM response. Unexpected format: {e}")
                rospy.logerr(f"Full response: {response.text}")
            except json.JSONDecodeError as e:
                rospy.logerr(f"Failed to decode the cleaned LLM JSON response: {e}")
                rospy.logerr(f"Cleaned response was: '{cleaned_response}'")
            except Exception as e:
                rospy.logerr(f"An unexpected error occurred: {e}")

            

if __name__ == '__main__':
    try:
        GeminiHanoiSolverNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass