# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement a **model-free Q-Learning control algorithm** using the
Gymnasium `FrozenLake-v1` environment.

The objective is to train an agent that learns the optimal action-value
function `Q(s,a)` through repeated interaction with the environment.

The agent should:

- Observe the current state.
- Select an action using an epsilon-greedy strategy.
- Receive a reward from the environment.
- Observe the next state.
- Update the Q-table using the Q-Learning update rule.
- Gradually reduce exploration during training.
- Learn a policy that attempts to reach the goal while avoiding holes.
- Display the final Q-table.
- Display the estimated state-value function.
- Display the learned policy.
- Plot the learning curve.
- Calculate the average reward obtained over the last 1000 episodes.

---


## Software Requirements

1. Python 3.x
2. Gymnasium
3. NumPy
4. Matplotlib
5. Jupyter Notebook 

## Environment Description
```

FrozenLake-v1 is a grid-world reinforcement learning environment provided by Gymnasium. The environment consists of a 4 × 4 grid with 16 states and 4 possible actions:

Action	Meaning
0	      Left
1	      Down
2	      Right
3	      Up
```

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

1. Initialize the FrozenLake-v1 environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros for all state-action pairs.
4. Set the learning rate \(\alpha\), discount factor \(\gamma\), initial epsilon, minimum epsilon, and epsilon decay rate.
5. Reset the environment at the beginning of each episode.
6. Select an action using the epsilon-greedy strategy.
7. Execute the selected action in the environment.
8. Observe the next state, reward, and termination status.
9. Calculate the Q-Learning target:

$$
Target = R+\gamma\max_a Q(S',a)
$$

10. Update the Q-value using:

$$
Q(S,A) \leftarrow Q(S,A)+
\alpha[Target-Q(S,A)]
$$

11. Move to the next state.
12. Continue until the episode terminates.
13. Store the reward obtained in the episode.
14. Reduce epsilon gradually to decrease exploration and increase exploitation.
15. Repeat the process for the specified number of episodes.
16. Obtain the state-value function using:

$$
V(S)=\max_a Q(S,a)
$$

17. Obtain the learned policy by selecting the action with the maximum Q-value for each state.
18. Plot the learning curve.
19. Calculate the average reward over the last 1000 episodes.

## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

episode_rewards = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        best_next_action = np.max(Q[next_state])
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_next_action - Q[state, action])
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", average_reward)

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()

```
---

## Output
![alt text](<Screenshot 2026-09-07 113405.png>)

![alt text](<Screenshot 2026-09-07 113412.png>)
---

## Result
The Q-Learning algorithm was successfully implemented in FrozenLake-v1, learning Q-values through repeated interaction using Q-table updates and an epsilon-greedy strategy. The learned Q-table produced the state-value function and greedy policy, while the learning curve and final average reward were used to evaluate performance.

---

## Inference

The experiment demonstrates that Q-Learning is a model-free control algorithm that learns a suitable policy through trial and error without requiring a predefined environment model. Through exploration and exploitation, the agent improves its Q-values and learns to reach the goal while avoiding holes in the stochastic FrozenLake environment.

---

