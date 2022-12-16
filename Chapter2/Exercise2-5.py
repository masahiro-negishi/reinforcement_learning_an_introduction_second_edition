# Experiments on the k-arms bandit problem
# The environment changes as time passes

import random
import numpy as np
import matplotlib.pyplot as plt

def KArmsBandit(k: int, max_steps: int, weighted_avg: bool, alpha: float = 0.1, sigma: float = 0.1):

    # Initialize q*(a) (E[R|A=a])
    q_star = [1.0 for i in range(k)]

    # Initialize Q_t(a) (The estimate for q*(a))
    Q_t = [0.0 for i in range(k)]
    N_t = [0 for i in range(k)] # The number of a particular action chosen

    # Reward
    reward_history = np.zeros(max_steps)
    optimal_history = np.zeros(max_steps, dtype=np.float64) 

    # Loop for max_steps times
    for t in range(max_steps):

        # Choose an action
        if(random.random() > sigma):
            # Greedy
            maxQ_t = max(Q_t)
            maxIndex = [i for i, x in enumerate(Q_t) if x == maxQ_t]
            a = maxIndex[random.randint(0, len(maxIndex)-1)]
        else:
            # Random
            a = random.randint(0, k-1)

        # Receive a reward
        reward = random.normalvariate(q_star[a], 1.0)

        # History
        reward_history[t] = reward
        maxq_star = max(q_star)
        OptimalIndex = [i for i, x in enumerate(q_star) if x == maxq_star]
        if(a in OptimalIndex):
            optimal_history[t] = 1

        # Update Q_t[a]
        if(weighted_avg):
            Q_t[a] += alpha * (reward - Q_t[a])
        else:
            N_t[a] += 1
            Q_t[a] += 1 / N_t[a] * (reward - Q_t[a])

        # Update q_star
        for i in range(k):
            q_star[i] += random.normalvariate(0, 0.01)
    
    return reward_history, optimal_history


def visualize():
    k = 10
    max_steps = 20000
    alpha = 0.1
    sigma = 0.1
    sample_mean_reward_hist_avg = np.zeros(max_steps, dtype=np.float64)
    weighted_avg_reward_hist_avg = np.zeros(max_steps, dtype=np.float64)
    sample_mean_optimal_hist_avg = np.zeros(max_steps, dtype=np.float64)
    weighted_avg_optimal_hist_avg = np.zeros(max_steps, dtype=np.float64)

    experiments_num = 2000
    for i in range(experiments_num):
        random.seed(i)
        sample_mean_reward_hist, sample_mean_optimal_hist = KArmsBandit(k, max_steps, False, sigma=sigma)
        sample_mean_reward_hist_avg += sample_mean_reward_hist
        sample_mean_optimal_hist_avg += sample_mean_optimal_hist
    for i in range(experiments_num):
        random.seed(i)
        weighted_avg_reward_hist, weighted_avg_optimal_hist = KArmsBandit(k, max_steps, True, alpha=alpha)
        weighted_avg_reward_hist_avg += weighted_avg_reward_hist
        weighted_avg_optimal_hist_avg += weighted_avg_optimal_hist

    
    sample_mean_reward_hist_avg /= experiments_num
    weighted_avg_reward_hist_avg /= experiments_num
    sample_mean_optimal_hist_avg /= experiments_num
    weighted_avg_optimal_hist_avg /= experiments_num

    x = [i+1 for i in range(max_steps)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{str(k)}-arms bandit problem under changing environments")
    ax.plot(x, sample_mean_reward_hist_avg, label="sample mean")
    ax.plot(x, weighted_avg_reward_hist_avg, label=f"weighted avg (alpha = {str(alpha)})")
    ax.set_xlabel("steps")
    ax.set_ylabel("average rewards")
    ax.legend()
    fig.savefig("Exercise2-5_reward.png")
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{str(k)}-arms bandit problem under changing environments")
    ax.plot(x, sample_mean_optimal_hist_avg, label="sample mean")
    ax.plot(x, weighted_avg_optimal_hist_avg, label=f"weighted avg (alpha = {str(alpha)})")
    ax.set_xlabel("steps")
    ax.set_ylabel("rate of optimal actions")
    ax.legend()
    fig.savefig("Exercise2-5_optimal.png")


if __name__=="__main__":
    visualize()