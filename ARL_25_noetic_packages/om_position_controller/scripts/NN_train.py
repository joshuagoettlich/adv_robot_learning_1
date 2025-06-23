#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# --- 1. Configuration ---
# These settings MUST match the configuration in your ROS node.

# Set the number of disks for the training data.
N_DISKS = 3

# Define all possible moves. The order is crucial and must be identical
# to the POSSIBLE_MOVES list in the ROS node.
POSSIBLE_MOVES = [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# The name of the model file to save. This must match MODEL_FILENAME in the ROS node.
MODEL_FILENAME = 'tower_of_hanoi_solver.h5'

# Global variable to hold the state of the pegs during data generation.
pegs = {}

# --- 2. Data Generation ---

def hanoi_solver_generator(total_disks, n, source, target, auxiliary):
    """
    A recursive generator that yields tuples of (current_state, optimal_move)
    for each step in the optimal solution of the Tower of Hanoi puzzle.
    
    Args:
        total_disks (int): The total number of disks in the puzzle.
        n (int): The number of disks to move in the current recursive step.
        source (str): The source peg ('A', 'B', or 'C').
        target (str): The target peg.
        auxiliary (str): The auxiliary peg.
    """
    if n > 0:
        # Step 1: Move n-1 disks from the source to the auxiliary peg.
        # This gets them out of the way.
        yield from hanoi_solver_generator(total_disks, n - 1, source, auxiliary, target)

        # Step 2: Yield the current state and the single move for the nth disk.
        # This is the core move at this stage of the recursion.
        state_before_move = get_current_state_vector(total_disks)
        move = (source, target)
        yield (state_before_move, move)

        # Update the global pegs state to reflect the move.
        disk_to_move = pegs[source].pop()
        pegs[target].append(disk_to_move)

        # Step 3: Move the n-1 disks from the auxiliary peg to the target peg.
        yield from hanoi_solver_generator(total_disks, n - 1, auxiliary, target, source)

def get_current_state_vector(n_disks):
    """
    Represents the current state of the pegs as a flat list (vector).
    This function's logic must perfectly match the `transform_state_to_vector`
    method in the ROS node.
    
    Args:
        n_disks (int): The total number of disks in the puzzle.
        
    Returns:
        list: A flat list representing the padded state of all pegs.
    """
    state_vector = []
    # Use a fixed order of pegs ('A', 'B', 'C') for consistent state representation.
    for peg_name in ['A', 'B', 'C']:
        peg_state = pegs.get(peg_name, [])[:]  # Make a copy
        # Pad the list with 0s to have a fixed-size input for the neural network.
        padded_peg = peg_state + [0] * (n_disks - len(peg_state))
        state_vector.extend(padded_peg)
    return state_vector

def generate_hanoi_training_data(n_disks):
    """
    Generates a complete dataset of (state, optimal_move) pairs.
    
    Args:
        n_disks (int): The number of disks to generate data for.
        
    Returns:
        (np.array, np.array): A tuple containing the training states (X) and labels (y).
    """
    global pegs
    # Initialize the pegs for the start of the puzzle.
    pegs = {
        'A': list(range(n_disks, 0, -1)),  # Disks are [3, 2, 1] for N=3
        'B': [],
        'C': []
    }
    
    # Create a mapping from a move tuple to an integer index (e.g., ('A', 'B') -> 0).
    move_to_index = {move: i for i, move in enumerate(POSSIBLE_MOVES)}
    
    states = []
    labels = []

    # The generator yields the state *before* each optimal move.
    for state_before_move, move in hanoi_solver_generator(n_disks, n_disks, 'A', 'C', 'B'):
        states.append(state_before_move)
        labels.append(move_to_index[move])

    return np.array(states), np.array(labels)

# --- 3. Model Definition and Training ---

if __name__ == '__main__':
    print(f"--- Tower of Hanoi Model Trainer ---")
    print(f"Generating training data for {N_DISKS} disks...")
    
    # Generate the dataset. For N_DISKS=3, this will be 2^3 - 1 = 7 samples.
    X_train, y_train = generate_hanoi_training_data(N_DISKS)
    print(f"Successfully generated {len(X_train)} training samples.")

    # One-hot encode the labels for categorical cross-entropy loss function.
    # e.g., label '2' becomes [0, 0, 1, 0, 0, 0]
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(POSSIBLE_MOVES))

    # --- Define the Neural Network Architecture ---
    print("\nBuilding the Neural Network model...")

    # The input shape is 3 pegs * N_DISKS positions per peg (3 * 3 = 9 for 3 disks).
    input_shape = (N_DISKS * 3,) 
    output_shape = len(POSSIBLE_MOVES) # 6 possible moves

    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape, name='input_layer'),
        Dropout(0.2), # Helps prevent overfitting
        Dense(128, activation='relu', name='hidden_layer_1'),
        Dropout(0.2),
        Dense(64, activation='relu', name='hidden_layer_2'),
        # Use 'softmax' for the output layer for multi-class classification.
        # It converts logits to probabilities that sum to 1.
        Dense(output_shape, activation='softmax', name='output_layer') 
    ])

    # Compile the model with an optimizer, loss function, and metrics.
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy', # Best loss function for one-hot encoded labels.
        metrics=['accuracy']
    )

    model.summary()

    # --- Train the Model ---
    print("\nTraining the model...")
    # For small datasets like this, we can train for many epochs.
    history = model.fit(
        X_train,
        y_train_one_hot,
        epochs=500,
        batch_size=8, # Use a small batch size for small datasets
        validation_split=0.2, # Use part of the data for validation
        verbose=1
    )

    # --- Evaluate and Save ---
    print("\nTraining finished. Evaluating model performance.")
    loss, accuracy = model.evaluate(X_train, y_train_one_hot, verbose=0)
    print(f"\nFinal Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Training Loss: {loss:.4f}")

    # Save the trained model to the specified file.
    if os.path.exists(MODEL_FILENAME):
        print(f"\nOverwriting existing model file '{MODEL_FILENAME}'.")
    else:
        print(f"\nSaving new model to '{MODEL_FILENAME}'.")
        
    model.save(MODEL_FILENAME)
    print("Model saved successfully. You can now use this file with your ROS node.")

