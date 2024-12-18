import numpy as np
import time

UP = np.array((-1, 0))
DOWN = np.array((1, 0))
LEFT = np.array((0, -1))
RIGHT = np.array((0, 1))

starting_position = np.array((2, 0))  # Starting position
ending_position = np.array((0, 3))
blocked_positions = np.array([(1, 1)])

# Define the grid
grid1 = np.array([
    [-1, -1  , -1, 100],  # -1 represents empty cells, 100 is the goal(ending position)
    [-1, None, -1, -1 ],  # None represents blocked cell
    [-1, -1  , -1, -1 ]
])

grid2 = np.array([
    [-3, -2  , -1, 100],  # -1 represents empty cells, 100 is the goal(ending position)
    [-4, None, -2, -1 ],  # None represents blocked cell
    [-5, -4  , -3, -2 ]
])

# Initialize Q-table
num_states = grid1.size
num_actions = 4
q_table = np.zeros((num_states, num_actions))

def reconstruct_path(starting_position, Q):
    state = starting_position
    path = [tuple(np.array(state))]
    

    while not np.array_equal(state, ending_position):  # Continue until reaching the goal state
        state_index = state_to_index(state)
        action = np.argmax(Q[state_index])  # Choose the action with the highest Q-value
        state = tuple(np.array(state) + np.array([UP, DOWN, LEFT, RIGHT][action]))
        path.append(state)
        
    return path

def state_to_index(state):
    return state[0]*grid1.shape[1] + state[1]

def index_to_state(index):
    return (index//grid1.shape[1], index%grid1.shape[1])

# Return possible actions
def get_available_actions(state):
    row, col = state
    available_actions = np.array([0, 1, 2, 3])

    if row == 0 or any(np.array_equal(state, underBlockedState) for underBlockedState in (blocked_positions + DOWN)):
        available_actions = available_actions[available_actions != 0] # Up
    if row == grid1.shape[0]-1 or any(np.array_equal(state, aboveBlockedState) for aboveBlockedState in (blocked_positions + UP)):
        available_actions = available_actions[available_actions != 1] # Down
    if col == 0 or any(np.array_equal(state, rightToBlockedState) for rightToBlockedState in (blocked_positions + RIGHT)):
        available_actions = available_actions[available_actions != 2] # Left
    if col == grid1.shape[1]-1 or any(np.array_equal(state, leftToBlockedState) for leftToBlockedState in (blocked_positions + LEFT)):
        available_actions = available_actions[available_actions != 3] # Right

    return available_actions

# Return best action
def get_best_action(state, epsilon):
    available_actions = get_available_actions(state)
    
    # Epsilon-greedy
    if np.random.rand() > epsilon:
        action = np.random.choice(available_actions)
    else:
        state_index = state_to_index(state)
        q_values = q_table[state_index, available_actions]
        max_q_value = np.max(q_values)
        best_actions = [action for action in available_actions if q_table[state_index, action] == max_q_value]
        action = np.random.choice(best_actions)
    
    # Comment next if and else for exercise 1a and 1b
    if not drunken_sailor():
        return action
    else:
        available_actions = available_actions[available_actions != action]
        return np.random.choice(available_actions)
    
    return action

def drunken_sailor(ratio=0.01):
    if np.random.rand() < ratio:
        return True
    else:
        return False

def q_learning(grid, alpha, gamma, epsilon, num_episodes = 4, convergence_threshold=0.0001, convergence_window=10):
    mean_q_value_changes = []
    print_first = False
    print_second = False
    for episode in range(num_episodes):
        state = starting_position  # Starting state
        prev_Q = np.copy(q_table)
        
        while not np.array_equal(state, ending_position):  # Until reaching the goal
            action = get_best_action(state, epsilon)

            next_state = np.array(state) + np.array([UP, DOWN, LEFT, RIGHT][action])  # Up Down Left Right
            
            # Reward function
            reward = grid[next_state[0], next_state[1]]

            state_index = state_to_index(state)
            next_state_index = state_to_index(next_state)
            q_table[state_index, action] = q_table[state_index, action] + alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])
            state = next_state
        
        mean_q_value_changes.append(np.mean(np.abs(q_table - prev_Q)))  
        
        # Check for convergence
        if len(mean_q_value_changes) > convergence_window:
            mean_change = np.mean(mean_q_value_changes[-convergence_window:])
            if not print_first:
                print("First Q-table:\n", q_table, '\n')
                print_first = True
            
            if mean_change < 0.001 and not print_second:
                print("Second Q-table:\n", q_table, '\n')
                print_second = True
                
            if mean_change < convergence_threshold:
                print("Converged in episode ", episode)
                break
            
    return q_table

# iterations
num_episodes = 500
alpha = 0.2  # Learning rate //   Tested with 0.1 0.2 0.3 0.4 0.5
gamma = 0.7  # Discount factor // Tested with 0.9 0.8 0.7 0.6 0.5
epsilon = 0.9 # Tested beetwen 0.1-0.9

start_time = time.time() # To calculate time

# Final Q-table
q_table = q_learning(grid1, alpha, gamma, epsilon, num_episodes) # Exercise 1a
#q_table = q_learning(grid2, alpha, gamma, epsilon, num_episodes) # Exercise 1b

elapsed_time = time.time() - start_time


print("Tiempo transcurrido:", elapsed_time, "segundos") # Time
print("Final Q-table:")
print("         UP         DOWN          LEFT        RIGHT")
print(q_table, '\n')

optimal_path = reconstruct_path(starting_position, q_table)
print("Optimal Path:", optimal_path)

"""
for index in range(len(q_table)):
    print(str(index_to_state(index)) + ": " + str(q_table[index]))
"""