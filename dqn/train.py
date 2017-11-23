import gym
import numpy as np

from dqn.agent import DQNAgent
from dqn.agent import EPISODES, EPISODE_LENGTH, BATCH_SIZE


environment_name = 'CartPole-v1'
environment = gym.make(environment_name)
environment.max_episode_steps = EPISODE_LENGTH

n_actions = environment.action_space.n
n_state_features = environment.observation_space.shape[0]

# Initialize DQN agent
agent = DQNAgent(n_state_features, n_actions)

for episode in range(EPISODES):

    state = environment.reset()
    state = np.reshape(state, [1, n_state_features])

    for t in range(EPISODE_LENGTH):

        # Predict next action using NN Value Function Approximation
        action = agent.get_action(state)

        # Interact with the environment and observe new state and reward
        next_state, reward, terminated, info = environment.step(action)

        # Huge negative reward if failed
        if terminated:
            reward = -100

        # Remember agent's experience: state / action / reward / next state
        next_state = np.reshape(next_state, [1, n_state_features])
        agent.remember(state, action, reward, next_state, terminated)

        # Change the current state
        state = next_state

        # Print statistics if agent failed and quit inner loop
        if terminated:
            print(f'Episode: {episode} of {EPISODES} (score: {t}s, exploration rate: {agent.epsilon:.4})')
            break

    # Re-train Value Function Approximation model if we have enough examples in memory
    if len(agent.memory) >= BATCH_SIZE:
        agent.experience_replay(BATCH_SIZE)

    # Save trained agent every once in a while
    if episode % 100 == 0:
        agent.save(f'./models/{environment_name}.h5')
