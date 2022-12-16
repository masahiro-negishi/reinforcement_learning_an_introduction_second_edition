# Experiments on the k-arms bandit problem
# The environment changes as time passes
# Compare four algorithms
# eps-greedy, constant-eps-greedy, constant-eps-greedy with optimistic initial value, UCB, gradient-bandit

import random
import numpy as np
import math
import matplotlib.pyplot as plt

def KArmsBandit(k: int, max_steps: int, algo: str, eps: float = 0.1, Q0: float = 1.0, c: float = 1.0, alpha: float = 0.1):
    """

    Args:
        k (int): the number of arms
        max_steps (int): max steps
        algo (str): "eps" or "constant" or "optimistic" or "ucb" or "gradient"
        eps (float): a parameter used when algo == "eps" or "constant"
        Q0 (float): a parameter used when algo == "optimistic"
        c (float): a parameter used when algo == "ucb"
        alpha (float): a parameter used when algo == "gradient"
    """

    # Initialize q*(a) (E[R|A=a])
    q_star = [1.0 for i in range(k)]

    # Initialize Q_t(a) (The estimate for q*(a))
    if(algo == "eps" or algo == "constant" or algo == "ucb"):
        Q_t = [0.0 for i in range(k)]
    elif(algo == "optimistic"):
        Q_t = [Q0 for i in range(k)]

    # The number of a particular action chosen
    if(algo == "eps" or algo == "ucb"):
        N_t = [0 for i in range(k)] 

    # H_t, reward_avg
    if(algo == "gradient"):
        H_t = [0.0 for i in range(k)]
        reward_avg = 0.0

    # Reward
    reward_sum = 0.0

    # Loop for max_steps times
    for t in range(max_steps):

        # Choose an action
        if(algo == "eps" or algo == "constant"):
            if(random.random() > eps):
                # Greedy
                maxQ_t = max(Q_t)
                maxIndex = [i for i, x in enumerate(Q_t) if x == maxQ_t]
                a = maxIndex[random.randint(0, len(maxIndex)-1)]
            else:
                # Random
                a = random.randint(0, k-1)
        elif(algo == "optimistic"):
            # Greedy
            maxQ_t = max(Q_t)
            maxIndex = [i for i, x in enumerate(Q_t) if x == maxQ_t]
            a = maxIndex[random.randint(0, len(maxIndex)-1)]
        elif(algo == "ucb"):
            nonzero_N_t = [max(nt, 0.00001) for nt in N_t]
            value = [qt + c * ((np.log(t+1) / nt)**0.5) for qt, nt in zip(Q_t, nonzero_N_t)]
            # Greedy
            maxval = max(value)
            maxIndex = [i for i, x in enumerate(value) if x == maxval]
            a = maxIndex[random.randint(0, len(maxIndex)-1)]
        elif(algo == "gradient"):
            maxH_t = max(H_t)
            e_H_t = [math.exp(ht - maxH_t) for ht in H_t]
            maxe_H_t = max(e_H_t)
            pi_t = [eht / maxe_H_t for eht in e_H_t]
            a = random.choices(list(range(k)), k=1, weights=pi_t)[0]

        # Receive a reward
        reward = random.normalvariate(q_star[a], 1.0)

        # History
        if(t >= max_steps / 2):
            reward_sum += reward

        # Update Q_t[a], N_t[a], H_t
        if(algo == "eps" or algo == "ucb"):
            N_t[a] += 1
            Q_t[a] += 1 / N_t[a] * (reward - Q_t[a])
        elif(algo == "constant" or algo == "optimistic"):
            Q_t[a] += 0.1 * (reward - Q_t[a])
        elif(algo == "gradient"):
            for i in range(k):
                if(i == a):
                    H_t[i] += alpha * (reward - reward_avg) * (1 - pi_t[i])
                else:
                    H_t[i] -= alpha * (reward - reward_avg) * pi_t[i]
            reward_avg = (reward_avg * t + reward) / (t + 1)

        # Update q_star
        for i in range(k):
            q_star[i] += random.normalvariate(0, 0.01)
    
    return reward_sum / (max_steps / 2)


def visualize():
    k = 10
    max_steps = 200000
    eps_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan])
    constant_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan])
    optimistic_avg = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    ucb_avg = np.array([np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gradient_avg = np.array([np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    experiments_num = 100

    for i, eps in enumerate([1/128, 1/64, 1/32, 1/16, 1/8, 1/4]):
        for e in range(experiments_num):
            random.seed(e)
            eps_avg[i] += KArmsBandit(k, max_steps, "eps", eps=eps)
            if (e+1) % 10 == 0:
                print(f"eps: {e+1} / {experiments_num}")
        eps_avg[i] / experiments_num
    
    for i, eps in enumerate([1/128, 1/64, 1/32, 1/16, 1/8, 1/4]):
        for e in range(experiments_num):
            random.seed(e)
            constant_avg[i] += KArmsBandit(k, max_steps, "constant", eps=eps)
            if (e+1) % 10 == 0:
                print(f"constant: {e+1} / {experiments_num}")
        constant_avg[i] / experiments_num

    for i, Q0 in enumerate([1/4, 1/2, 1, 2, 4]):
        for e in range(experiments_num):
            random.seed(e)
            optimistic_avg[i+5] += KArmsBandit(k, max_steps, "optimistic", Q0=Q0)
            if (e+1) % 10 == 0:
                print(f"optimistic: {e+1} / {experiments_num}")
        optimistic_avg[i] / experiments_num

    for i, c in enumerate([1/16, 1/8, 1/4, 1/2, 1, 2, 4]):
        for e in range(experiments_num):
            random.seed(e)
            ucb_avg[i+3] += KArmsBandit(k, max_steps, "ucb", c=c)
            if (e+1) % 10 == 0:
                print(f"ucb: {e+1} / {experiments_num}")
        ucb_avg[i] / experiments_num

    for i, alpha in enumerate([1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]):
        for e in range(experiments_num):
            random.seed(e)
            gradient_avg[i+2] += KArmsBandit(k, max_steps, "gradient", alpha=alpha)
            if (e+1) % 10 == 0:
                print(f"gradient: {e+1} / {experiments_num}")
        gradient_avg[i] / experiments_num

    x = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{str(k)}-arms bandit problem under changing environments")
    ax.plot(x, eps_avg, label="eps-greedy (eps)")
    ax.plot(x, constant_avg, label="eps-greedy with constant step size (= 0.1) (eps)")
    ax.plot(x, optimistic_avg, label="greedy with optimistic initial value (step size = 0.1) (Q0)")
    ax.plot(x, ucb_avg, label="UCB (c)")
    ax.plot(x, gradient_avg, label="gradient (alpha)")
    ax.set_xlabel("eps, alpha, c, Q0")
    ax.set_ylabel("average rewards of latter 100000 steps")
    ax.set_xscale('log')
    ax.set_xticks([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4])
    ax.set_xticklabels(["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    ax.legend(loc="upper right")
    fig.savefig("Exercise2-11.png")


if __name__=="__main__":
    visualize()