# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
for fixed epsilon
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

# Define the missing hyperparameter used in generate_episode
max_steps_per_episode = 100

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate a complete episode
    episode = generate_episode(epsilon)

    G = 0
    visited = set()

    # Calculate returns backwards
    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        # First-visit Monte Carlo
        if (state, action) not in visited:

            visited.add((state, action))

            # Update Q-value
            Q[state, action] += alpha * (
                G - Q[state, action]
            )

    # Store episode reward
    total_reward = sum(
        step[2] for step in episode
    )

    episode_rewards.append(total_reward)

   

print("Monte Carlo Control Training Completed!")
print("Final Epsilon:", epsilon)



```

---

## Output
Final Q-table:


for fixed epsilon

<img width="296" height="382" alt="image" src="https://github.com/user-attachments/assets/e5b0ba36-7630-456b-92f5-bb285286065d" />


for variable epsilon

<img width="367" height="387" alt="image" src="https://github.com/user-attachments/assets/e45c265b-6fe6-4f49-9c6d-02fcbd3acfad" />

Estimated State-Value Function:


for fixed epsilon

<img width="430" height="176" alt="image" src="https://github.com/user-attachments/assets/134d15db-92b6-4802-b422-39aed1e65b00" />



for variable epsilon  

<img width="402" height="160" alt="image" src="https://github.com/user-attachments/assets/0339347c-bf28-4b9f-a99e-c65844e5edba" />


Learned Policy:


for fixed epsilon

<img width="242" height="135" alt="image" src="https://github.com/user-attachments/assets/6f9b0c89-8bc4-4002-8ede-e9d9a923420f" />



for variable epsilon

<img width="527" height="152" alt="image" src="https://github.com/user-attachments/assets/c54856d1-f219-4f52-8646-8e19430690f7" />



Average reward over last 1000 episodes:


for fixed epsilon

<img width="923" height="626" alt="image" src="https://github.com/user-attachments/assets/8cb2e7a0-9126-48ce-a187-3334f29b883e" />




for variable epsilon

<img width="891" height="597" alt="image" src="https://github.com/user-attachments/assets/15c83df1-6dc2-4904-9591-b92e3dd1266e" />
 
```
---

## Result
```text



```
---

## Inference
```text



```





---

