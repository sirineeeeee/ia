import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    max_q_sprime = np.max(Q[sprime])
    Q[s, a] += alpha * (r + gamma * max_q_sprime - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilon):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if np.random.rand() < epsilon:
        action = np.random.choice(Q.shape[1])
    else: 
        action = np.argmax(Q[s])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1  
    gamma = 0.9  
    epsilon = 0.1

    n_epochs = 2000 
    max_itr_per_epoch = 500 
    rewards = []

    for e in range(1900):
        r = 0
        S, info = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

            S = Sprime
            epsilon = max(0.1, epsilon * 0.99)
            if done:
                break

        rewards.append(r)
        print("Episode #", e + 1, " : Reward =", r)

    print("Average reward for the first 500 episodes:", np.mean(rewards))

    env.close()

    env = gym.make("Taxi-v3", render_mode="human")

    for e in range(1900, n_epochs):
        r = 0
        S, info = env.reset()
        env.render()
        epsilon=0
        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

            S = Sprime
            env.render()

            if done:
                break

        rewards.append(r)
        print("Episode #", e + 1, " : Reward =", r)

    print("Average reward over all episodes:", np.mean(rewards))

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.show()

    print("Training finished.\n")
    
    env.close()
