import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt

#######################################
# 1. Initialize transition and reward matrices
# 2. Fill transition and reward matrices with correct values
#######################################

#states and actions initialized
num_states = 5
num_actions = 4

# Create transition and reward matrices below:

# Your code here
T = np.zeros((num_states, num_states, num_actions))

# Your code here
R = np.zeros((num_states, num_states, num_actions))


# Set the entries in T and R below as per the MDP used in the assignment diagram:
'''
YOUR CODE HERE
'''

#p, q, s, r values initialized
p = 0.5
q = 0.5
s = 0.5
r = 1

#transition matrices
T[0,0,0] = p
T[2,0,1] = q
T[4,2,3] = s
T[1,0,0] = 1-p
T[1,0,1] = 1-q
T[0,2,3] = 1-s
T[2,1,2] = r/3
T[3,1,2] = r/3
T[4,1,2] = r/3

#reward matrices
R[4,1,2] = 10
R[4,2,3] = 10

#######################################
# 3. Map indices to action-labels 
#######################################

A = {0: "A1", 1: "A2", 2: "A3", 3: "A4", 4: "A5"} # map each index to an action-label, such as "A1", "A2", etc.


#######################################
# Initialize the gymnasium environment
#######################################

# No change required in this section

P_0 = np.array([1, 0, 0, 0, 0])    # This is simply the initial probability distribution which denotes where your agent is, i.e. the start state.

env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R)


#######################################
# 4. Random exploration
#######################################

'''
First, we reset the environment and get the initial observation.

The observation tells you which state you are in - in this case, indices 0-4 map to states S1 - S5.

Since we set P_0 = [1, 0, 0, 0, 0], the initial state is always S1 after env.reset() is called.
'''

observation, info = env.reset()

'''
Below, write the code for random exploration, i.e. randomly choosing an action at each time-step and executing it.

A random action is simply a random integer between 0 and the number of actions (num_actions not inclusive).
However, you should make sure that the chosen action can actually be taken from the current state.
If it is not a legal move, generate a new random move.

Avoid hardcoding actions even for states where there is only one action available. That way, your
code is more general, and may be easily adapted to a different environment later.

You will use the following line of code to explore at each time step:

observation, reward, terminated, truncated, info = env.step(action)

The above line of code is used to take one step in the environment using the chosen action.
It takes as input the action chosen by the agent, and returns the next observation (i.e., state),
reward, whether the episode terminated (terminal states), whether the episode was 
truncated (max iterations reached), and additional information.

If at any point the episode is terminated (this happens when we reach a terminal state, 
and the env.step() function returns True for terminated), you should
end the episode in order to reset the environment, and start a new one.

Keep track of the total reward in each episode, and reset the environment when the episode terminates.

Print the average reward obtained over 10000 episodes. 

'''

'''
YOUR CODE HERE
'''
#initializes number of episodes and reward
num_episodes = 10000
total_reward = 0

#resets after every episode
for episode in range (num_episodes):
    observation, info = env.reset()
    current_reward = 0

    while True:
        # no actions means no iterations
        if num_actions == 0:
            break
        else:
            # based on the action, run the environment and add the reward for that action
            chosen_action = np.random.choice(num_actions)
            #if action isn't valid
            while (np.sum(T[:, observation, chosen_action]) == 0):
                #choose a new action
                chosen_action = np.random.randint(num_actions)
            #gets values from env.step using chosen action
            observation, reward, terminated, truncated, info = env.step(chosen_action)
            #add reward to current eps's rewards
            current_reward += reward
            # if the environment is terminated, end the program
            if terminated:
                break
        #adds current eps's reward to total reward
        total_reward += current_reward

#calculates average rewards
avg_reward = total_reward/num_episodes

print("Average reward obtained: ", avg_reward)

#######################################
# 5. Policy evaluation 
# 6. Plotting V_pi(s)
#######################################

gamma = 0.9

'''
Initialize the value function V(s) = 0 for all states s.
Use the Bellman update to iteratively update the value function, given the following policy:

S1 -> A1, S2 -> A3, S3 -> A4

Plot the value of S1 over time (i.e., at each iteration of Bellman update).
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the value function of S1 and S2 after 100 iterations.

'''

'''
YOUR CODE HERE for first policy
'''
#initializes policy and states
policy_one = {0:0, 1:2, 2:3}
states = [0, 1, 2]

#initializes iterations and list
num_iterations = 100
S1_values_one = []

#initializes matrices to be used
V = np.zeros(num_states)
V_fill = np.zeros(5)
v_copy = np.copy(V_fill)

#iterates through all iterations in num_iterations
for iteration in range(num_iterations):
    #appends to first policy s1 value list
    S1_values_one.append(v_copy[0])
    #iterates through the states and sets the actions for each state
    for s in states:
        chosen_action = policy_one[s]
        v_copy[s] = np.sum(T[:, s, chosen_action] * (R[:, s, chosen_action] + gamma * V_fill))
        # current_value = new_value1
        V_fill = v_copy

#plots the values of S1 over time
plt.plot(range(num_iterations), S1_values_one)

#labels for plot
plt.xlabel("Iterations")
plt.ylabel("Value of S1 for Policy One")

#plt.savefig() to save the plot
plt.savefig("s1_over_time_policy_one.png")


#######################################
# 7. Evaluating a Second Policy
#######################################

'''
Now change the policy to:

S1 -> A2, S2 -> A3, S3 -> A4

Re-run Bellman updates for all states.

Plot the value of S1 over time (i.e., at each iteration of Bellman update). 
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the value function of S1 and S2 after 100 iterations.
'''
print("Value of S1 and S2 after 100 iterations for first policy: ", V[0], V[1])

'''
YOUR CODE HERE for second policy
'''
plt.clf()

#initializes policy and states
policy_two = {0:1, 1:2, 2:3}
states = [0, 1, 2]

#initializes iterations and list
num_iterations = 100
S1_values_two = []

#Matrices
V = np.zeros(num_states)
V_fill = np.zeros(5)
v_copy = np.copy(V_fill)

#iterates through all iterations in num_iterations
for iteration in range(num_iterations):
    #appends to first policy s1 value list
    S1_values_two.append(v_copy[0])

    #iterates through the states and sets the actions for each state
    for s in states:
        chosen_action = policy_two[s]
        v_copy[s] = np.sum(T[:, s, chosen_action] * (R[:, s, chosen_action] + gamma * V_fill))
        # current_value = new_value1
        V_fill = v_copy

#plots the values of S1 over time
plt.plot(range(num_iterations), S1_values_two)

#labels for plot
plt.xlabel("Iterations")
plt.ylabel("Value of S1 for Policy Two")

#plt.savefig() to save the plot
plt.savefig("s1_over_time_policy_two.png")

print("Value of S1 and S2 after 100 iterations for second policy: ", V[0], V[1])


#######################################
# 8. Value Iteration for Best Policy
# 9. Output Best Policy
#######################################

'''
Initialize the value function V(s) = 0 for all states s.

Use value iteration to find the optimal policy for the MDP.

Plot V_opt(S1) over time (i.e., at each iteration of Bellman update).
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the optimal policy after 100 iterations.

'''
'''
YOUR CODE HERE
'''
#clears plot before plotting again
plt.clf()
#initializes policy and states and matrice
policy = {}
states = [0, 1, 2]
V = np.zeros(num_states)

#initializes iterations and list
S1_values_Opt_V = []
num_iterations = 100

#iterates through all iterations in num_iterations
for iteration in range(num_iterations):
    S1_values_Opt_V.append(V[0])
    # copy of V
    v_copy = V.copy()
    #iterates through the possible states
    for s in states:
        # state values for actions
        opt_values = []
        #iterates through the possible actions for the chosen state
        for chosen_action in range(num_actions):
            opt_value = 0
            #iterates through the possible values of s_prime for the chosen action
            for s_prime in range(num_states):
                #check equation
                opt_value += np.sum(T[s_prime, s, chosen_action] * (R[s_prime, s, chosen_action] + gamma * V[s_prime]))
            #add opt value for this iteration to the opt values list
            opt_values.append(opt_value)        
        # get the sums from all the sprimes and then compare them to their corresponding actions 
        # finds the optimal value
        v_copy[s] = max(opt_values)
        policy[s] = np.argmax(opt_values)
        #reassigns to V after loop
        V = v_copy.copy()

#plots the values of S1 over time
plt.plot(range(num_iterations), S1_values_Opt_V)

#labels for plot
plt.xlabel("Iterations")
plt.ylabel("Value of S1 for Optimal Policy")

#plt.savefig() to save the plot
plt.savefig("s1_over_time_optimal_policy.png")

#changed based on professors comments
print("Optimal policy: ", [A[policy[i]] for i in policy])



































