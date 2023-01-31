import random
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import csv
import math


def _look_and_leap(data: List[float], stopping_index: int, max_in_list: int):
    max_val = max(data[0:stopping_index]) if stopping_index>0 else 0
    # i will help decided when to take less than optimal data due to number of remaining data points
    for i ,(x) in enumerate(data[stopping_index :-1]): 
        progress = (i + stopping_index)/len(data)
        if x >= max_val:
            return x
        elif progress > 0.5 and abs(x - max_val) < max_in_list*.02:
            return x
        elif progress > 0.75 and abs(x - max_val) < max_in_list*.05:
            return x
        elif progress > 0.9 and abs(x - max_val) < max_in_list*.1:
            return x
        elif progress > 0.97 and abs(x - max_val) < max_in_list*.2:
            return x

    return data[-1]

def test_data(data: List[List[float]], upper_bound: int) -> List[float]:
    # find the optimmal stopping place by using the loop and leak function
    length_of_data = len(data[0])
    overall_stats = [0 for _ in range(length_of_data)]
    number_of_lists = len(data)
    # i represents the index in which we stop looking
    for i in range(1, length_of_data): 
        # iterating through each data set in the list
        for data_set in data:
            max_value = max(data_set)
            max_found = _look_and_leap(data_set, i, upper_bound)
            
            if max_found == max_value:
                overall_stats[i] += 1
    overall_stats = [x/number_of_lists*100 for x in overall_stats]
    return overall_stats

def test_random_data(data_points: int, number_of_lists: int, upper_bound_of_integers: int):
    # multitude of ways of generating random data
    # data = [np.random.randint(upper_bound_of_integers, size=(data_points)) for i in range(number_of_lists)]
    data = []
    for _ in range(number_of_lists):
        # data.append([random.random()*upper_bound_of_integers for _ in range(data_points)])
        data.append([np.random.normal(50,10)*upper_bound_of_integers for _ in range(data_points)])
    return test_data(data, upper_bound_of_integers)

def _read_file(name: str):
    data = []
    with open(name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for val in reader:
            data.append(int(val[0]))
    numpy_array = np.array(data)
    return numpy_array

def run_scenarios():
    # set up and run the given csv files
    scenario_1 = _read_file('scenario1.csv')
    scenario_2 = _read_file('scenario2.csv')
    s1_results = test_data([scenario_1], 99)
    run_37_rule(scenario_1, 99)
    s2_results = test_data([scenario_2], 100)
    run_37_rule(scenario_2, 100)
    percent_normalizer = scenario_1.size/100
    x_values = [i/percent_normalizer for i in range(scenario_1.size)]
    
    plt.plot(x_values, s1_results, label='Scenario 1')
    plt.plot(x_values, s2_results, label='Scenario 2')
    plt.legend()
    plt.show()

def run_37_rule(data: List[float], upper_bound: int):
    index_37 = math.ceil(.37*len(data))
    val = _look_and_leap(data, index_37, upper_bound)
    print(f'Max Value found using 37% rule: {val}')

def run_random():
    # set up and run the random sets of data
    data_points = 100
    lists = 1000
    upper_bound = 1000
    percent_normalizer = data_points/100
    random_results = test_random_data(data_points, lists, upper_bound)
    x_values = [i/percent_normalizer for i in range(data_points)]
    plt.plot(x_values, random_results, label = 'Random Set of Lists')
    plt.axvline(x = .37*data_points/percent_normalizer, color = 'k', label = '37 percent')
    plt.xlabel('Percent of data searched')
    plt.ylabel('Success rate')
    plt.show()

def capitalism(data: List[np.ndarray]) -> List[float]:
    # I use this function to show the average value from each position taking into account the penalty for searching
    overal_dist = []
    for i in range(1, data[0].size):
        value_distributiion = []
        for dist in data:
            dist = list(dist)
            max_val = _look_and_leap_with_penalty(dist, i, max(dist))
            value_distributiion.append(max_val)
        overal_dist.append(sum(value_distributiion)/len(value_distributiion))
    return overal_dist
    
def _look_and_leap_with_penalty(data: List[float], stopping_index: int, max_in_list: int):
    max_val = 0
    for i, (x) in enumerate(data[0:stopping_index]):
        if (x - i) >= max_val:
            max_val = x - i
    # i will help decided when to take less than optimal data due to number of remaining data points
    for i ,(x) in enumerate(data[stopping_index :-1]): 
        i += stopping_index
        progress = (i + stopping_index)/len(data)
        if (x - i) >= max_val:
            return x - i
        elif progress > 0.5 and abs(x - max_val - i) < max_in_list*.1:
            return x - i
        elif progress > 0.75 and abs(x - max_val - i) < max_in_list*.2:
            return x - i
        elif progress > 0.9 and abs(x - max_val - i) < max_in_list*.3:
            return x - i
        elif progress > 0.97 and abs(x - max_val - i) < max_in_list*.4:
            return x - i

    return data[-1]


def run_captitalism():
    # find the optimal stopping point when there is a penalty
    uniform_dist = [np.random.uniform(1,99,99) for _ in range(1000)]
    normal_dist = [np.random.normal(50,10,99) for _ in range(1000)]
    data = capitalism(normal_dist)
    max_norm = max(data)
    index_norm = data.index(max_norm) 
    print(f'The maximum average value found from the normal distribution is {max_norm} at index {index_norm}')
    data2 = capitalism(uniform_dist)
    max_uni = max(data2)
    index_uni = data2.index(max_uni)
    print(f'The maximum average value found from the uniform distribution is {max_uni} at index {index_uni}')
    plt.plot(range(0,98), data, label='normal')
    plt.plot(range(0,98), data2, label='uniform')
    plt.legend()
    plt.show()

def run_37_with_penalty(data: List[float], upper_bound: int):
    index_37 = math.ceil(.37*len(data))
    max_val = _look_and_leap_with_penalty(data, index_37, max(data))
    return max_val


def capitlasism_37():
    uniform = np.random.uniform(1,99,99)
    normal = np.random.normal(50,10,99)
    uni_max = run_37_with_penalty(uniform, max(uniform))
    norm_max = run_37_with_penalty(normal, max(normal))
    print(f"Uniform: max using 37% rule: {uni_max}")
    print(f'Normal: max using 37% rule: {norm_max}')


if __name__ == "__main__":
    np.random.seed(50)
    run_scenarios()
    run_random()
    run_captitalism()
    capitlasism_37()

