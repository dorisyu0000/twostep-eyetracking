import numpy as np
import random
import json
import os
from itertools import product

# Define the range of numbers to be used for combinations
number_range = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# Generate all possible combinations for the nodes that contribute to path ratings
combinations = list(product(number_range, repeat=7))  # Adjusted for the number of contributing nodes

def calculate_difficulty(combination):
    # Extract the ratings for each path
    root_1, root_2, root_3, chidl_11,child_21, child_31, child_32 = combination

    # Calculate the sum of the ratings for each path influenced by these nodes
    path1 = root_1 + chidl_11
    path2 = root_2 + child_21
    path3 = root_3 + child_31
    path4 = root_3 + child_32

    # Determine the primary measure of trial difficulty
    paths = [path1, path2, path3, path4]
    max_value = max(paths)
    sum_value = sum(paths)
    difficulty = max_value - ((sum_value - max_value) / (len(paths) - 1))
    round_difficulty = round(difficulty, 1)
    if difficulty - round_difficulty == 0.0:
        return round_difficulty
    else:
        return None

# Apply the difficulty calculation to each combination
difficulties = [calculate_difficulty(combination) for combination in combinations]

# Categorize combinations by difficulty
categorized_combinations = {}
for i, combination in enumerate(combinations):
    difficulty = difficulties[i]
    if difficulty is not None:
        if difficulty not in categorized_combinations:
            categorized_combinations[difficulty] = []
        categorized_combinations[difficulty].append(combination)

# Filter for significant number of trials per difficulty level
for difficulty, trials in categorized_combinations.items():
    if len(trials) > 100:
        categorized_combinations[difficulty] = random.sample(trials, 100)

# Convert the categorized combinations into JSON format
json_data_categorized = json.dumps(categorized_combinations, indent=4)

# Ensure the directory exists and write the file
dest = "static/json/"
os.makedirs(dest, exist_ok=True)
with open(os.path.join(dest, "trials_3.json"), "w") as file:
    file.write(json_data_categorized)

# Print the counts of difficulties
difficulty_counts = {difficulty: len(trials) for difficulty, trials in categorized_combinations.items()}
print(difficulty_counts)
