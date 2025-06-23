#!/root/catkin_ws/gemini_env/bin/python3

import rospy
import json
import requests

# Import the standard String message type.
from std_msgs.msg import String
from std_msgs.msg import Char
import yaml

try:
    with open('api_key.yaml', 'r') as file:
        config = yaml.safe_load(file)
        api_key = config['api_key']

    # Now you can use the api_key in your application
    print(f"Successfully loaded API key: {api_key}")

except FileNotFoundError:
    print("Error: 'api_key.yaml' not found. Please create the file.")
except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")
except KeyError:
    print("Error: 'api_key' not found in the YAML file.")


SYSTEM_PROMPT = """
You are a master strategist for a robotic arm that solves the Tower of Hanoi puzzle. Your task is to determine the single next optimal move.

**Rules of the Game:**
1.  You can only move one disk at a time.
2.  A larger disk can never be placed on top of a smaller disk.
3.  You can only move the topmost disk from one of the three pegs (A, B, C) to another.

**State Validation:**
Before determining a move, you MUST first validate that the provided 'Current State' is legal. If any larger disk is on top of a smaller disk, the state is invalid.

**Reasoning Process:**
1. First, validate the input 'Current State' as per the 'State Validation' rule.
2. If valid, identify the top disk on each peg.
3. Internally list all possible moves (e.g., A->B, A->C, B->A, etc.).
4. For each possible move, check if it is legal according to the rule 'a larger disk cannot be placed on a smaller disk'.
5. From the list of legal moves, select the single most optimal move that follows the standard algorithm to solve the puzzle in the minimum number of steps.
6. Finally, provide ONLY the selected optimal move in the required JSON format.

**Output Format:**
- If the state is **valid**, your response MUST be a JSON object with the source and destination peg. Example: `{"source_peg": "A", "destination_peg": "C"}`
- If the state is **invalid**, your response MUST be a JSON object with a single "error" key. Example: `{"error": "Invalid state on Peg A: disk 2 cannot be on disk 1."}`

Do not include any other text or explanations.
"""

class HanoiLogicNode:
    def __init__(self):
        rospy.init_node('hanoi_llm_robot_logic_node', anonymous=True)
        self.api_key = api_key  # Load the API key from the YAML file
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
        HanoiLogicNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass