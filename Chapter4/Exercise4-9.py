import matplotlib.pyplot as plt
import argparse

def ValueIteration(ph, theta):
    # Initialization
    V = [0.0 for i in range(101)]
    
    # Iteration
    while(1):
        max_diff = 0.0
        for s in range(1, 100):
            next_v = 0
            for a in range(0, min(s, 100-s)+1):
                if(s+a == 100):
                    q = ph * (1 + V[s+a]) + (1 - ph) * V[s-a]
                else:
                    q = ph * V[s+a] + (1 - ph) * V[s-a]
                next_v = max(next_v, q)
            max_diff = max(max_diff, abs(next_v - V[s]))
            V[s] = next_v
        print(max_diff)
        if(max_diff < theta):
            break
    # Optimal policy
    pi = [0 for i in range(101)] # pi[0] and pi[100] should be ignored
    for s in range(1, 100):
        max_a = -1
        max_q = -1.0
        for a in range(0, min(s, 100-s)+1):
            if(s+a == 100):
                q = ph * (1 + V[s+a]) + (1 - ph) * V[s-a]
            else:
                q = ph * V[s+a] + (1 - ph) * V[s-a]
            if(q >= max_q):
                max_q = q
                max_a = a
        pi[s] = max_a
    
    # Visualize state value and optimal policy
    x = [i for i in range(1, 100)]
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Estimated State Value")
    ax.plot(x, V[1:100])
    ax.set_xlabel("Capital")
    ax.set_ylabel("Estimated State Value")
    fig.savefig(f"Exercise4-9_estimated_state_value_ph={ph}_theta={theta}.png")
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Final policy")
    ax.plot(x, pi[1:100])
    ax.set_xlabel("Capital")
    ax.set_ylabel("Final Policy")
    fig.savefig(f"Exercise4-9_final_policy_ph={ph}_theta={theta}.png")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ph", type=float, help="The probability of a coin flip tuning up heads")
    parser.add_argument("--theta", type=float, help="Tolerance for convergence")
    args = parser.parse_args()
    print(args.ph, args.theta)
    ValueIteration(args.ph, args.theta)
