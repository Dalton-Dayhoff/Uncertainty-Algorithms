from typing import List
import numpy as np
from matplotlib import pyplot as plt
import statistics

def generate_random_data(number_of_lists: int, points_per_list: int, upper_bound: int ) -> List[np.ndarray]:
    data = [np.random.randint(upper_bound, size=(points_per_list)) for i in range(number_of_lists)]
    return data

def look_and_leap(data: np.ndarray, stopping_index: int):
    max = 0
    for i, (x) in enumerate(data):
        if (i > stopping_index) and (x > max):
            return x
        if x > max:
            max = x
        elif i == (data.size - 1):
            return x

def test_random_data(data_points: int) -> List[float]:
    data = generate_random_data(1000, data_points, 100)
    overall_stats = []
    for i in range(data[0].size):
        individual_stats = []
        print("i = ",i)
        for data_set in data:
            max_found = look_and_leap(data_set, i)
            max_value = max(data_set)
            
            if max_found == max_value:
                individual_stats.append(1)
            else:
                individual_stats.append(0)
        overall_stats.append(sum(individual_stats)/len(individual_stats))
    return overall_stats


if __name__ == "__main__":
    number_of_data_points = 100
    data = test_random_data(number_of_data_points)
    x_values = range(0,number_of_data_points)
    plt.plot(x_values, data)
    plt.show()
