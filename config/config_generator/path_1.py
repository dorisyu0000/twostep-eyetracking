import networkx as nx
import numpy as np
import random as random
import json
import os
import matplotlib.pyplot as plt
from itertools import product
import json

# Define the range of numbers to be used for combinations
number_range = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# Generate all possible combinations of 5 numbers from the given range
combinations = list(product(number_range, repeat=5))

# Prepare the data to be saved as JSON
json_data = json.dumps(combinations)

def calculate_difficulty(combination):
    # Extract the ratings for each path
    left_top, right_top, left_bottom_left, left_bottom_right, right_bottom_left = combination
    # Calculate the sum of the ratings for each path
    value_leftleft = left_top + left_bottom_left
    value_leftright = left_top + left_bottom_right
    value_rightleft = right_top + right_bottom_left
    # Calculate the average value of the two other paths for each path
    max_value = max(value_leftleft, value_leftright, value_rightleft)
    sum_value = value_leftleft + value_leftright + value_rightleft 
    
    # Determine the primary measure of trial difficulty
    difficulty = max_value - ((sum_value - max_value)/2)
    # If difficulty in the range of any integer +- 0.25, round to the nearest integer, else exclude the combination
    round_difficulty = round(difficulty)
    if difficulty - round_difficulty == 0.0:
        return round_difficulty*1.0
    else:
        return None




# Apply the difficulty calculation to each combination
difficulties = [calculate_difficulty(combination) for combination in combinations]

# Categorize combinations by difficulty, excluding 0 difficulty combinations and rounding 0.5 up
categorized_combinations = {}
categorized_combinations_excluding_equal_values = {}

for i, combination in enumerate(combinations):
    left_top, right_top, left_bottom_left, left_bottom_right,  right_bottom_left  = combination
    value_leftleft = left_top + left_bottom_left
    value_leftright = left_top + left_bottom_right
    value_rightleft = right_top + right_bottom_left
    # value_rightright = right_top + right_bottom_right
    
    # Exclude combinations where any value of any path is equal
    if value_leftleft == value_leftright or value_leftleft == value_rightleft  or value_leftright == value_rightleft:
        continue
    
    difficulty = difficulties[i]
    if difficulty == 0.0:  # Skip combinations with 0 difficulty
        continue

    if difficulty not in categorized_combinations_excluding_equal_values:
        categorized_combinations_excluding_equal_values[difficulty] = []
    categorized_combinations_excluding_equal_values[difficulty].append(combination)
    #for difficulty form 2.0 to 11.0, limit the number of combinations for each difficulty to 100
    
for difficulty, trials in categorized_combinations_excluding_equal_values.items():
    if difficulty is not None and 2.0 <= difficulty <= 11.0:
        if len(trials) >= 100:
            categorized_combinations_excluding_equal_values[difficulty] = random.sample(trials, 100)
        
    
difficulty_counts_excluding_equal_values = {difficulty: len(combinations) for difficulty, combinations in categorized_combinations_excluding_equal_values.items()}
print(difficulty_counts_excluding_equal_values)

# Convert the categorized combinations into JSON format
json_data_categorized = json.dumps(categorized_combinations_excluding_equal_values, indent=4)

dest = "static/json/config/"
os.makedirs(dest, exist_ok=True)

# Write the categorized combinations to a JSON file
with open(f"trials_1.json", "w") as file:
    file.write(json_data_categorized)

"static/json/combinations_1.json"