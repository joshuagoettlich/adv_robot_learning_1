#!/usr/bin/env python3

import rospy
import json
import time
import requests
import yaml
import os

# System prompt for the LLM
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
"""

class HanoiLogicNode:
    def __init__(self):
        rospy.init_node('hanoi_llm_robot_logic_node', anonymous=True)

        # --- Load API Key from YAML file ---
        self.api_key = self.load_api_key_from_yaml()
        if not self.api_key:
            rospy.logerr("API KEY NOT FOUND or failed to load. Shutting down.")
            return
            
        rospy.loginfo("API Key loaded successfully.")
        # ------------------------------------

        rospy.Subscriber('/hanoi_game_state', HanoiGameState, self.game_state_callback)
        
        # NOTE: Original robot publishers/subscribers would go here.
        # They are commented out for clarity as they are not needed for the logic test.
        # self.joint_pubs = { ... }
        # self.current_joint_states = {}
        # rospy.Subscriber('/open_manipulator_6dof/joint_states', JointState, self._joint_states_callback)
        
        rospy.loginfo("Hanoi LLM Logic Node initialized and subscribing to /hanoi_game_state.")

    def load_api_key_from_yaml(self):
        """Loads the API key from a YAML file named api_key.yaml."""
        try:
            # Assumes the yaml file is in the same directory as the script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            yaml_path = os.path.join(script_dir, 'api_key.yaml')
            
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                if config and 'api_key' in config:
                    return config['api_key']
                else:
                    rospy.logerr(f"API key not found in {yaml_path}")
                    return None
        except FileNotFoundError:
            rospy.logerr(f"API key config file not found at {yaml_path}")
            return None
        except yaml.YAMLError as e:
            rospy.logerr(f"Error parsing YAML file: {e}")
            return None
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred while loading the API key: {e}")
            return None

    def game_state_callback(self, msg):
        rospy.loginfo("--- Received Game State ---")
        peg_a = tuple(msg.peg_a_disks)
        peg_b = tuple(msg.peg_b_disks)
        peg_c = tuple(msg.peg_c_disks)
        rospy.loginfo(f"Peg A: {peg_a}")
        rospy.loginfo(f"Peg B: {peg_b}")
        rospy.loginfo(f"Peg C: {peg_c}")
        rospy.loginfo("-------------------------")

        if not peg_a and not peg_b:
            rospy.loginfo("Puzzle Solved! No further moves needed.")
            return

        current_state = f"Peg A: {peg_a}, Peg B: {peg_b}, Peg C: {peg_c}"
        full_prompt = f"{SYSTEM_PROMPT}\n\nCurrent State:\n{current_state}"
        
        rospy.loginfo("Asking LLM for the next move via API call...")
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": full_prompt}]}]}

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=20)
            response.raise_for_status() 

            result_json = response.json()
            llm_text_response = result_json['candidates'][0]['content']['parts'][0]['text']
            
            # Clean the response to ensure it's valid JSON
            cleaned_response = llm_text_response.strip().replace('```json', '').replace('```', '').strip()
            move_data = json.loads(cleaned_response)

            if 'error' in move_data:
                rospy.logwarn(f"LLM reported an invalid state: {move_data['error']}")
                return 

            source = move_data['source_peg']
            destination = move_data['destination_peg']
            
            rospy.loginfo(f"LLM Decided: Move from Peg {source} to Peg {destination}")
            self.execute_llm_move(source, destination)

        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Network error calling API: {e}")
        except (KeyError, IndexError, TypeError) as e:
            rospy.logerr(f"Failed to parse LLM response. Unexpected format: {e}")
            rospy.logerr(f"Full response received: {response.text}")
        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to decode JSON from LLM response: {e}")
            rospy.logerr(f"Cleaned response text was: {cleaned_response}")
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred: {e}")

    def execute_llm_move(self, source_peg, dest_peg):
        """Placeholder for robot motion control."""
        rospy.loginfo(f"--- EXECUTING ROBOT ACTION: {source_peg} -> {dest_peg} ---")
        # In a real scenario, this is where you would call your robot's
        # motion planning and execution functions.
        rospy.loginfo("Robot move sequence placeholder complete.")

if __name__ == '__main__':
    try:
        node = HanoiLogicNode()
        if node.api_key: # Only spin if initialization was successful
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
