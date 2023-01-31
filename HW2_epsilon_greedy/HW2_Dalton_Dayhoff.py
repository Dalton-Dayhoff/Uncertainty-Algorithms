import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import copy

def get_probabilities(drift=0):
    probs = [
        np.random.normal(0, 5),  #0
        np.random.normal(-0.5,12), #1
        np.random.normal(2,3.9), #2
        np.random.normal(-0.5,7), #3
        np.random.normal(-1.2,8), #4
        np.random.normal(-3,7), #5
        np.random.normal(-10,20), #6
        np.random.normal(-0.5,1), #7
        np.random.normal(-1,2), #8
        np.random.normal(1,6), #9
        np.random.normal(0.7,4), #10
        np.random.normal(-6,11), #11
        np.random.normal(-7,1), #12
        np.random.normal(-0.5,2), #13
        np.random.normal(-6.5,1), #14
        np.random.normal(-3,6), #15
        np.random.normal(0,8), #16
        np.random.normal(2,3.9), #17
        np.random.normal(-9,12), #18
        np.random.normal(-1,6), #19
        np.random.normal(-4.5,8) #20           
    ]
    
    return probs

def choice(epsilon: float, values: List[float], averages: List[float]):
    if random.random() > epsilon:
        val = max(averages)
        index = averages.index(val)
        return values[index], index
    else:
        val = random.choice(values)
        index = values.index(val)
        return val, index

def convergance(epsilon: float, iterations: int) -> List[float]:
    drift = 0

    current_values = get_probabilities(drift)
    all_values = [[] for _ in current_values]
    averages = [0 for _ in current_values]
    all_averages = []
    for i in range(iterations):
        val, ind = choice(epsilon, current_values, averages)
        all_values[ind].append(val)
        averages[ind] = sum(all_values[ind])/len(all_values[ind])
        all_averages.append(copy.deepcopy(averages))
        current_values = get_probabilities(drift)
    max_val = [max(reward) for reward in all_averages]
    max_index = [reward.index(max(reward)) for reward in all_averages]
    reordered_averages = []
    for i in range(len(averages)):
        values = []
        for j in range(len(all_averages)):
            values.append(all_averages[j][i])
        reordered_averages.append(values)
    return reordered_averages, max_val, max_index


def plot_convergance(averages: List[float], max_val: float, index: float, epsilon: float):
    x_vals = [i for i in range(len(averages[0]))]
    maxes = []
    for i, (reward_over_time) in enumerate(averages):
        plt.plot(x_vals, reward_over_time, label=f"Arm {i}")
        maxes.append(max(reward_over_time))
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -.05), ncol=(len(averages)/4))
    plt.title(f'Convergance for epsion: {epsilon}')
    plt.show()


def epsilons(iterations=10000, eps = [.01, .05, .1, .4], plot = True) -> List[List[int]]:
    convergance_per_epsilon = []
    maxes = []
    indecies = []
    for ep in eps:
        con, max_val, index = convergance(ep, iterations)
        convergance_per_epsilon.append(con)
        maxes.append(max_val)
        indecies.append(index)
    if plot:
        for i, (averages) in enumerate(convergance_per_epsilon):
            plot_convergance(averages, maxes[i][-1], indecies[i][-1], eps[i])
    return indecies

def find_optimal_epsilon():
    time_steps = 5000
    iterations = 5000
    # first iteration
    test_epsilons = np.linspace(0.01, 0.5, 20)
    indices_by_epsilon: List[List[float]] = [[] for _ in range(len(test_epsilons))]
    for _ in range(iterations):
        indicies = epsilons(time_steps, test_epsilons, False)
        for i, (indices_per_epsilon) in enumerate(indicies): 
            indices_by_epsilon[i].append(indices_per_epsilon)
    # rearrange data to get average correct choice at each time
    rearranged_data = [0 for _ in range(len(test_epsilons))]
    for i, (i_epsilon) in enumerate(indices_by_epsilon):
        for index in i_epsilon:
            if index == 2 or index == 17:
                rearranged_data[i] += 1
    averages = [sums/iterations for sums in rearranged_data]
    plt.plot(test_epsilons, averages)
    plt.show()







if __name__ == "__main__":
    np.random.seed(48)
    # epsilons()
    find_optimal_epsilon()
