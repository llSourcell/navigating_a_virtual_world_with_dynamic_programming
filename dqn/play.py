import gym
import numpy as np

from dqn.agent import DQNAgent
from dqn.agent import EPISODES, EPISODE_LENGTH

environment_name = 'CartPole-v1'
environment = gym.make(environment_name)
environment.max_episode_steps = EPISODE_LENGTH

n_actions = environment.action_space.n
n_state_features = environment.observation_space.shape[0]

# Initialize DQN agent
agent = DQNAgent(n_state_features, n_actions, epsilon=0.0)

# Load pre-trained agent
agent.load(f'./models/{environment_name}.h5')

for episode in range(EPISODES):

    state = environment.reset()
    state = np.reshape(state, [1, n_state_features])

    for t in range(EPISODE_LENGTH):

        # Visualize environment
        environment.render()

        # Predict next action using NN Value Function Approximation
        action = agent.get_action(state)

        # Interact with the environment and observe new state and reward
        next_state, reward, terminated, info = environment.step(action)
        next_state = np.reshape(next_state, [1, n_state_features])

        # Change the current state
        state = next_state

        # Print statistics if agent failed and quit inner loop
        if terminated:
            print(f'Episode: {episode} of {EPISODES} (score: {t}s, exploration rate: {agent.epsilon:.4})')
            break
