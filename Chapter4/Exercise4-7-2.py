# Program for example 4.7

import numpy as np
import math

def Initialize():
    pi = [[0 for j in range(21)] for i in range(21)]
    V = [[0.0 for j in range(21)] for i in range(21)]
    return pi, V


def Preparation():
    poisson_2 = [0.0 for i in range(21)]
    poisson_3 = [0.0 for i in range(21)]
    poisson_4 = [0.0 for i in range(21)]
    poisson_cum_2 = [0.0 for i in range(21)]
    poisson_cum_3 = [0.0 for i in range(21)]
    poisson_cum_4 = [0.0 for i in range(21)]
    for i in range(21):
        poisson_2[i] = (2**i/math.factorial(i)) * math.exp(-2)
        poisson_3[i] = (3**i/math.factorial(i)) * math.exp(-3)
        poisson_4[i] = (4**i/math.factorial(i)) * math.exp(-4)
    for i in range(21):
        poisson_cum_2[i] = poisson_2[i]
        poisson_cum_3[i] = poisson_3[i]
        poisson_cum_4[i] = poisson_4[i]
        if(i > 0):
            poisson_cum_2[i] += poisson_cum_2[i-1]
            poisson_cum_3[i] += poisson_cum_3[i-1]
            poisson_cum_4[i] += poisson_cum_4[i-1]
    return poisson_2, poisson_3, poisson_4, poisson_cum_2, poisson_cum_3, poisson_cum_4
    

def PolicyEvaluation(pi: list, V: list, theta: float, poisson_2: list, poisson_3: list, poisson_4: list, 
                    poisson_cum_2: list, poisson_cum_3: list, poisson_cum_4: list):
    """ pi --> V
    """
    iter_count = 1
    while(1):
        max_diff = 0.0
        for i in range(21):
            for j in range(21):
                # S = (i, j)
                pre_v = V[i][j]
                daytime_S = [i - pi[i][j], j + pi[i][j]]
                if(daytime_S[0] > 10 and daytime_S[1] > 10):
                    penalty = 8
                elif(daytime_S[0] <= 10 and daytime_S[1] <= 10):
                    penalty = 0
                else:
                    penalty = 4
                updated_v = 0.0
                for rent1 in range(daytime_S[0]+1):
                    for return1 in range(21+rent1-daytime_S[0]):
                        for rent2 in range(daytime_S[1]+1):
                            for return2 in range(21+rent2-daytime_S[1]):
                                p = 1.0
                                # rent1
                                if(rent1 < daytime_S[0]):
                                    p *= poisson_3[rent1]
                                else:
                                    if(daytime_S[0] != 0):
                                        p *= 1 - poisson_cum_3[daytime_S[0]-1]
                                # return1
                                if(return1 < 20+rent1-daytime_S[0]):
                                    p *= poisson_3[return1]
                                else:
                                    if(20+rent1-daytime_S[0] != 0):
                                        p *= 1 - poisson_cum_3[20+rent1-daytime_S[0]-1]
                                # rent2
                                if(rent2 < daytime_S[1]):
                                    p *= poisson_4[rent2]
                                else:
                                    if(daytime_S[1] != 0):
                                        p *= 1 - poisson_cum_4[daytime_S[1]-1]
                                # return2
                                if(return2 < 20+rent2-daytime_S[1]):
                                    p *= poisson_2[return2]
                                else:
                                    if(20+rent2-daytime_S[1] != 0):
                                        p *= 1 - poisson_cum_2[20+rent2-daytime_S[1]-1]
                                # r, s'
                                if(pi[i][j] > 0):
                                    r = (rent1 + rent2) * 10 - 2 * abs(pi[i][j] - 1) - penalty
                                else:
                                    r = (rent1 + rent2) * 10 - 2 * abs(pi[i][j]) - penalty
                                next_s = (daytime_S[0] - rent1 + return1, daytime_S[1] - rent2 + return2)
                                updated_v += p * (r + 0.9 * V[next_s[0]][next_s[1]])
                V[i][j] = updated_v
                max_diff = max(max_diff, updated_v - pre_v)
        if(max_diff < theta):
            break
        iter_count += 1
    print(f"# of iteration: {iter_count}")
    return V


def PolicyImprovement(pi: list, V:list, poisson_2: list, poisson_3: list, poisson_4: list, 
                    poisson_cum_2: list, poisson_cum_3: list, poisson_cum_4: list):
    """V --> pi
    """
    policy_stable = True
    updated_count = 0
    for i in range(21):
        for j in range(21):
            pre_a = pi[i][j]
            new_a = -1
            max_pi = -1.0
            for a in range(max(-5, -j, i-20), min(5, i, 20-j)+1):
                daytime_S = [i - a, j + a]
                if(daytime_S[0] > 10 and daytime_S[1] > 10):
                    penalty = 8
                elif(daytime_S[0] <= 10 and daytime_S[1] <= 10):
                    penalty = 0
                else:
                    penalty = 4
                q_s_a = 0.0
                for rent1 in range(daytime_S[0]+1):
                    for return1 in range(21+rent1-daytime_S[0]):
                        for rent2 in range(daytime_S[1]+1):
                            for return2 in range(21+rent2-daytime_S[1]):
                                p = 1.0
                                # rent1
                                if(rent1 < daytime_S[0]):
                                    p *= poisson_3[rent1]
                                else:
                                    if(daytime_S[0] != 0):
                                        p *= 1 - poisson_cum_3[daytime_S[0]-1]
                                # return1
                                if(return1 < 20+rent1-daytime_S[0]):
                                    p *= poisson_3[return1]
                                else:
                                    if(20+rent1-daytime_S[0] != 0):
                                        p *= 1 - poisson_cum_3[20+rent1-daytime_S[0]-1]
                                # rent2
                                if(rent2 < daytime_S[1]):
                                    p *= poisson_4[rent2]
                                else:
                                    if(daytime_S[1] != 0):
                                        p *= 1 - poisson_cum_4[daytime_S[1]-1]
                                # return2
                                if(return2 < 20+rent2-daytime_S[1]):
                                    p *= poisson_2[return2]
                                else:
                                    if(20+rent2-daytime_S[1] != 0):
                                        p *= 1 - poisson_cum_2[20+rent2-daytime_S[1]-1]
                                # r, s'
                                if(a > 0):
                                    r = (rent1 + rent2) * 10 - 2 * abs(a - 1) - penalty
                                else:
                                    r = (rent1 + rent2) * 10 - 2 * abs(a) - penalty
                                next_s = (daytime_S[0] - rent1 + return1, daytime_S[1] - rent2 + return2)
                                q_s_a += p * (r + 0.9 * V[next_s[0]][next_s[1]])
                if(max_pi < q_s_a):
                    max_pi = q_s_a
                    new_a = a
            pi[i][j] = new_a
            if(pre_a != new_a):
                policy_stable = False
                updated_count += 1
    print(f"{updated_count} / 441 are updated")
    return pi, policy_stable


def PolicyIteration():
    theta = 1e-4

    pi, V = Initialize()
    poisson_2, poisson_3, poisson_4, poisson_cum_2, poisson_cum_3, poisson_cum_4 = Preparation()

    iter_count = 1
    while(1):
        print(f"Iter {iter_count}:")
        print("Policy Evaluation")
        V = PolicyEvaluation(pi, V, theta, poisson_2, poisson_3, poisson_4, poisson_cum_2, poisson_cum_3, poisson_cum_4)
        print("Policy Improvement")
        pi, policy_stable = PolicyImprovement(pi, V, poisson_2, poisson_3, poisson_4, poisson_cum_2, poisson_cum_3, poisson_cum_4)
        if(policy_stable):
            break
        iter_count += 1

    for i in range(21):
        print("[ ", end="")
        for j in range(21):
            print(pi[i][j], end="")
            if(j != 21):
                if(pi[i][j] < 0):
                    print(" ", end="")
                else:
                    print("  ", end="")
        print("]")
    

if __name__=="__main__":
    PolicyIteration()
