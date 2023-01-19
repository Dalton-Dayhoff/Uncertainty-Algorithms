from typing import List
import numpy as np
from matplotlib import pyplot as plt
import csv


def _look_and_leap(data: np.ndarray, stopping_index: int):
    max = 0
    for i, (x) in enumerate(data):
        if (i > stopping_index) and (x >= max):
            return x
        if x > max:
            max = x
        elif i == (data.size - 1):
            return x

def test_data(data: List[np.ndarray]) -> List[float]:
    overall_stats = []
    for i in range(data[0].size): #data[0].size gives the number of data points per list
        individual_stats = []
        print("i = ",i)
        for data_set in data:
            max_found = _look_and_leap(data_set, i)
            max_value = max(data_set)
            
            if max_found == max_value:
                individual_stats.append(1)
            else:
                individual_stats.append(0)
        overall_stats.append(sum(individual_stats)/len(individual_stats) * 100)
    return overall_stats

def test_random_data(data_points: int, number_of_lists: int, upper_bound_of_integers: int):
    data = [np.random.randint(upper_bound_of_integers, size=(data_points)) for i in range(number_of_lists)]
    return test_data(data)

def read_file(name: str):
    data = []
    with open(name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for val in reader:
            data.append(int(val[0]))
    numpy_array = np.array(data)
    return numpy_array

def run_scenarios():
    scenario_1 = read_file('scenario1.csv')
    scenario_2 = read_file('scenario2.csv')
    s1_results = test_data([scenario_1])
    s2_results = test_data([scenario_2])
    percent_normalizer = scenario_1.size/100
    x_values = [i/percent_normalizer for i in range(scenario_1.size)]
    plt.plot(x_values, s1_results)
    plt.plot(x_values, s2_results)
    plt.show()

if __name__ == "__main__":
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

