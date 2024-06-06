import networkx as nx
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
import scipy
from scipy.stats import rv_discrete

# Create a Python dictionary with None as a value
data = {
    'key': None
}

IMAGES = [
    "pattern_1.png",
    "pattern_2.png",
    "pattern_3.png",
    "pattern_4.png",
    "pattern_5.png",
    "pattern_6.png",
    "pattern_7.png",
    "pattern_8.png",
    "pattern_9.png"
]


def regular_tree(branching):
    """
    Generates a regular tree where each level has a specified number of branches.
    """
    tree = []
    
    def rec(d):
        children = []
        tree.append(children)
        idx = len(tree) - 1
        if d < len(branching):
            for i in range(branching[d]):
                child = rec(d + 1)
                children.append(child)
        return idx

    rec(0)
    return tree

def empty_tree():
    return [[]]

def tree_join(g1, g2):
    """
    Joins two trees by adding a new root node.
    """
    n1 = len(g1)
    g1 = [[y + 1 for y in x] for x in g1]
    g2 = [[y + 1 + n1 for y in x] for x in g2]
    return [[2, n1 + 2]] + g1 + g2

def random_tree(splits):
    if splits == 0:
        return empty_tree()
    if splits == 1:
        return tree_join(empty_tree(), empty_tree())

    left = random.randint(0, splits - 1)
    right = splits - 1 - left
    return tree_join(random_tree(left), random_tree(right))

def valid_reward(n,rdist):
    """
    Ensures that the reward distribution has enough elements for sampling.
    If the list of rewards is shorter than n, it extends the list with additional elements.
    """
    while len(rdist.x) < n:
        rdist.x.append(0)

def paths(problem):
    """
    Returns all paths in the problem graph.
    """
    graph = problem['graph']
    start = problem['start']
    def rec(node):
        if node == []:
            return [[]]
        else:
            paths = []
            for child in node:
                child_paths = rec(graph[child])
                for path in child_paths:
                    paths.append([child] + path)
            return paths
    return rec(graph[start])

def calculate_path_rewards(graph, rewards, start):
    """Calculate the sum of rewards for each path in the graph."""
    path_rewards = {}
    n = len(graph)  # Number of nodes in the graph
    if len(rewards) < n:
        raise ValueError("The length of rewards must be at least as large as the number of nodes in the graph")

    for node, children in enumerate(graph):
        if node == start or node >= n:
            continue  # Skip the start node and nodes outside the range
        for child in children:
            if child == start or child >= n:
                continue
            path = tuple(sorted([node, child]))
            path_rewards[path] = rewards[node] + rewards[child]
    return path_rewards

def sample_requirement(rewards, graph, start, rdist, max_attempts=1000):
    """Iteratively resample rewards until each path has a unique sum or maximum attempts are reached."""
    for _ in range(max_attempts):
        path_rewards = calculate_path_rewards(graph, rewards, start)
        if len(set(path_rewards.values())) == len(path_rewards):
            # Unique path rewards found
            return rewards
        rewards = rdist.rand()

    # If max_attempts are reached without finding unique path rewards
    return None  # or raise an exception

# def sample_graph(n,base=None):
#     if base is None:
#         base = [[1, 2], [3, 4], [5, 6]]
#     base.extend([[] for i in range(n-len(base))])
#     perm = random.sample(range(len(base)), len(base))
#     graph = []
#     for idx in perm:
#         graph.append([perm.index(i) for i in base[idx] if i != []])
#     start = perm.index(0)
#     return graph, perm, start

def sample_graph(n, base = None):
    if base is None:
        base = [[1, 2], [3, 4], [5, 6]]
    graph = base + [[] for _ in range(n - len(base))]  # Extend the graph to n nodes
    start = 0  # Assuming start is the first node in the provided base structure
    perm = list(range(n))
    random.shuffle(perm)  # Shuffle indices to simulate graph shuffling
    
    # Apply permutation to shuffle the graph structure
    shuffled_graph = [None] * n
    for i in range(n):
        if graph[i]:
            shuffled_graph[perm[i]] = [perm[j] for j in graph[i]]
        else:
            shuffled_graph[perm[i]] = []
    
    # Adjust start based on shuffled indices
    shuffled_start = perm[start]
    
    return shuffled_graph, perm, shuffled_start

class Distribution:
    def rand(self):
        difficulties = np.arange(2, 12, 1)
        return np.random.choice(difficulties)

class Distribution_2:
    def rand(self):
        difficulties = np.arange(2, 12, 1)
        return np.random.choice(difficulties)

def sample_problem_1(n, trialNumber=None, rewards = None, n_steps=-1, graph=None, start=None, rdist = None, difficulty_distribution=Distribution()):
   # Load combinations from the categorized JSON file
    difficulty = difficulty_distribution.rand()
    difficulty_key = f"{difficulty:.1f}"

    with open("config/config_generator/trials_1.json", "r") as file:
        categorized_combinations = json.load(file)
    
    if difficulty_key in categorized_combinations:
        selected_combinations = categorized_combinations[difficulty_key]
        selected_combination = random.choice(selected_combinations)
    else:
        raise ValueError(f"No combination found for the sampled difficulty {difficulty_key}")

    num_random_rewards_needed = n - 1 - len(selected_combination)
    random_rewards = random.choices([-1,-2,-3,-4,1,2,3,4,0], k=num_random_rewards_needed)
    
    initial_rewards = [None] + selected_combination + random_rewards
    
    # Sample and shuffle the graph, keeping track of the permutation used
    graph, perm, start = sample_graph(n, base=[[1, 2], [3, 4], [5], [], [], [], [7, 8], [8, 9], [], []])
    
    # Apply the same permutation to shuffle the initial rewards
    shuffled_rewards = [None] * n
    for i, reward in enumerate(initial_rewards):
        shuffled_rewards[perm[i]] = reward

    if trialNumber is None:
        trialNumber = 0

    return {'graph': graph, 'rewards': shuffled_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber, 'difficulty': difficulty_key}

def find_leaf_indices(graph):
    # Example implementation; adjust according to your graph structure
    leaf_indices = [i for i, nodes in enumerate(graph) if not nodes]
    return leaf_indices

def shuffle_rewards_with_graph(rewards, perm, start):
    # Example shuffling logic, keeping start node's reward as None
    shuffled_rewards = [None if i == start else rewards[perm.index(i)] for i in range(len(rewards))]
    return shuffled_rewards

def sample_problem_2(n, trialNumber=None, n_steps=-1, graph=None, start=None, rdist = None, difficulty_distribution=Distribution()):
   # Load combinations from the categorized JSON file
    difficulty = difficulty_distribution.rand()
    difficulty_key = f"{difficulty:.1f}"

    with open("config/config_generator/trials_2.json", "r") as file:
        categorized_combinations = json.load(file)
    
    if difficulty_key in categorized_combinations:
        selected_combinations = categorized_combinations[difficulty_key]
        selected_combination = random.choice(selected_combinations)
    else:
        raise ValueError(f"No combination found for the sampled difficulty {difficulty_key}")

    num_random_rewards_needed = n - 1 - len(selected_combination)
    random_rewards = random.choices([-1,-2,-3,-4,1,2,3,4,0], k=num_random_rewards_needed)
    
    initial_rewards = [None] + selected_combination + random_rewards
    
    # Sample and shuffle the graph, keeping track of the permutation used
    graph, perm, start = sample_graph(n, base = [[1, 2],[3, 4],[5,6],[],[],[],[],[8,9],[9],[]])
    
    # Apply the same permutation to shuffle the initial rewards
    shuffled_rewards = [None] * n
    for i, reward in enumerate(initial_rewards):
        shuffled_rewards[perm[i]] = reward

    if trialNumber is None:
        trialNumber = 0

    return {'graph': graph, 'rewards': shuffled_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber,'difficulty': difficulty_key}




# def sample_problem_3(n, trialNumber=None, n_steps=-1, graph=None, start=None, rdist=None):


#     possible_combinations_6 = [
#         [1, 1, 3, 2, -3, 0], [3, -3, 2, -1, 0, 2], [-1, -2, 4, -1, 1, 2],
#         [-1, -2, 2, 0, -3, -2], [4, 1, 3, -3, -2, 0], [-4, -4, 4, -2, 0, -3]
#     ]
    
#     possible_combinations_5 = [
#         [1, 1, 3, 2, -3], [3, -3, 2, -1, 0], [-1, -2, 4, -1, 1],
#         [-1, -2, 2, 0, -3], [4, 1, 3, -3, -2], [-4, -4, 4, -2, 0]
#     ]

#     reward_count = 6 if trialNumber % 50 == 0 else 5

#     if reward_count == 6:
#         available_combinations = [combo for combo in possible_combinations_6]
#         base = [[1, 2], [3, 4], [5, 6], [], [], [], [], [8, 9], [9], []]
#     else:
#         available_combinations = [combo for combo in possible_combinations_5 ]
#         base = [[1, 2], [3, 4], [5], [], [], [], [7, 8], [8, 9], [], []]


#     reward_combination = random.choice(available_combinations)
   
#     num_random_rewards_needed = n - 1 - reward_count
#     random_rewards = random.choices([-1, -2, -3, -4, 1, 2, 3, 4, 0], k=num_random_rewards_needed)
#     initial_rewards = [None] + reward_combination + random_rewards

#     graph, perm, start = sample_graph(n, base)  # Ensure sample_graph is implemented as needed

#     shuffled_rewards = [None] * n
#     for i, reward in enumerate(initial_rewards):
#         shuffled_rewards[perm[i]] = reward

#     return {'graph': graph, 'rewards': shuffled_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber, 'difficulty': 20.0}





def sample_problem_3(n, trialNumber=None, n_steps=-1, graph=None, start=None, rdist = None, difficulty_distribution=Distribution()):
    difficulty = difficulty_distribution.rand()
    difficulty_key = f"{difficulty:.1f}"

    with open("config/config_generator/trials_3.json", "r") as file:
        categorized_combinations = json.load(file)
    
    if difficulty_key in categorized_combinations:
        selected_combinations = categorized_combinations[difficulty_key]
        selected_combination = random.choice(selected_combinations)
    else:
        raise ValueError(f"No combination found for the sampled difficulty {difficulty_key}")

    num_random_rewards_needed = n - 1 - len(selected_combination)
    random_rewards = random.choices([-1,-2,-3,-4,1,2,3,4,0], k=num_random_rewards_needed)
    
    initial_rewards = [None] + selected_combination + random_rewards
    
    # Sample and shuffle the graph, keeping track of the permutation used
    graph, perm, start = sample_graph(n, base = [[1, 2, 3], [4, 5], [6], [7], [], [], [], [], [9], []])
    
    # Apply the same permutation to shuffle the initial rewards
    shuffled_rewards = [None] * n
    for i, reward in enumerate(initial_rewards):
        shuffled_rewards[perm[i]] = reward

    if trialNumber is None:
        trialNumber = 0

    return {'graph': graph, 'rewards': shuffled_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber,'difficulty': difficulty_key}

def sample_problem_4(n, trialNumber=None, n_steps=-1, graph=None, start=None, rdist = None, difficulty_distribution=Distribution()):
    difficulty = difficulty_distribution.rand()
    difficulty_key = f"{difficulty:.1f}"

    with open("config/config_generator/trials_4.json", "r") as file:
        categorized_combinations = json.load(file)
    
    if difficulty_key in categorized_combinations:
        selected_combinations = categorized_combinations[difficulty_key]
        selected_combination = random.choice(selected_combinations)
    else:
        raise ValueError(f"No combination found for the sampled difficulty {difficulty_key}")

    num_random_rewards_needed = n - 1 - len(selected_combination)
    random_rewards = random.choices([-1,-2,-3,-4,1,2,3,4,0], k=num_random_rewards_needed)
    
    initial_rewards = [None] + selected_combination + random_rewards
    
    # Sample and shuffle the graph, keeping track of the permutation used
    graph, perm, start = sample_graph(n, base = [[1, 2, 3], [4, 5], [6, 7], [8], [], [], [], [], [], []])
    
    # Apply the same permutation to shuffle the initial rewards
    shuffled_rewards = [None] * n
    for i, reward in enumerate(initial_rewards):
        shuffled_rewards[perm[i]] = reward

    if trialNumber is None:
        trialNumber = 0

    return {'graph': graph, 'rewards': shuffled_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber,'difficulty': difficulty_key}


def sample_practice(n, trialNumber = None, n_steps=-1, rdist=None, rewards=None, graph=None, start=None):
    if graph is None:
        graph, perm, start = sample_graph(n)
    else:
        perm = list(range(n))
    if rewards is None and rdist is not None:
        rewards = rdist.rand()
    if trialNumber is None:
        trialNumber = 0

    # rewards = sample_requirement(rewards, graph, start, rdist)

    all_rewards = [None] * n
     # Assign rewards to non-leaf nodes, excluding the start node
    non_leaf_nodes = set()
    for node, children in enumerate(graph):
        if children and node != start:  # Exclude start node
            non_leaf_nodes.add(node)
        for child in children:
            if child != start:  # Exclude start node
                non_leaf_nodes.add(child)
    # Distribute rewards among non-leaf, non-start nodes
    for rewards, node in zip(rewards, non_leaf_nodes):
        all_rewards[node] = rewards
    return {'graph': graph, 'rewards': all_rewards, 'start': start, 'n_steps': n_steps, 'trialNumber': trialNumber}


def learn_reward(n, graph=None, start=None):
    # Define the reward combinations
    reward_combinations = [
        (0, -4), (0, 4), (-4, 4), (-2, 2), (2, 4), (4, 2), (0, 2), (0, -2),
        (-1, 2), (-3, 2), (-1, 0), (1, 0), (0, 3), (0, -3), (0, 4), (0, -4),
        (1, -3), (1, -4), (3, -2),(-4,-1),(-3,0),(-1,4),(3,-1)
    ]

    # Sample a graph if none is provided
    if graph is None:
        base = [[1, 2]]  # This needs to be defined properly for your use case
        graph, perm, start = sample_graph(n, base)
        # Ensuring that the start node has at least two children
        if len(graph[start]) < 2:
            raise ValueError("The start node must have at least two children.")
    else:
        perm = list(range(n))  # No permutation if graph is already given

    # Sample rewards from the given combinations
    rewards = random.choice(reward_combinations)

    # Initialize all rewards to None
    all_rewards = [None] * n

    # Assign rewards to the first two children of the start node
    if len(graph[start]) >= 2:
        all_rewards[graph[start][0]] = rewards[0]
        all_rewards[graph[start][1]] = rewards[1]

    return {'graph': graph, 'rewards': all_rewards, 'start': start}


# Example usage
# Assuming functions like states, paths, and sample_graph are defined
# problem = sample_problem(n=5)
# print(result)

def sample_problem(**kwargs):
    for i in range(10000):
        problem = sample_problem_1(**kwargs)
        return problem
   
def discrete_uniform(v):
    probs = np.ones(len(v)) / len(v)
    return np.random.choice(v, p=probs)

def linear_rewards(n):
    assert n % 2 == 0
    n2 = n // 2
    return list(range(-n2, 0)) + list(range(1, n2 + 1))

class Shuffler:
    def __init__(self, x):
        self.x = x
    def rand(self):
        random.shuffle(self.x)
        return self.x

class IIDSampler:
    def __init__(self, n, x):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.x = x

    def rand(self):
        return random.choices(self.x, k=self.n)
    

import networkx as nx

def intro_graph(n):
    g = []
    for i in range(n):
        g.append([(i + 1)%n,(i + 3)%n])
    return g

def intro_problem(n, n_steps=-1, rdist=None, rewards=None, graph=None, start=None):
    if graph is None:
        graph = intro_graph(n)
    if rewards is None:
        if rdist is not None:
            rewards = rdist.rand()
        else:
            rewards = [None] * n  # Default to a list of zeros
    if len(rewards) < n:
        rewards.extend([None] * (n - len(rewards)))
    elif len(rewards) > n:
        rewards = rewards[:n]  
    random.shuffle(rewards)
    return {'graph': graph, 'rewards': rewards, 'start': start if start is not None else 0, 'n_steps': n_steps}

    
    

def make_trials():
    n = 10
    rewards = [1,2,3,4,-1,-2,-3,-4,0]
    rdist = IIDSampler(n, rewards) 
    kws = {'n': n, 'rdist': rdist}
    trial_sets = []

    for _ in range(10):
        problem = learn_reward(n)
        trial_sets.append(problem)  

    main = [] 
    trialNumber = 1


    for _ in range(200):  
        if trialNumber % 2 == 0:
            problem_1 = sample_problem_1(**kws, trialNumber=trialNumber)
            main.append(problem_1)
        elif trialNumber % 2 == 1:
            problem_2 = sample_problem_2(**kws, trialNumber=trialNumber)
            main.append(problem_2)
        # elif trialNumber % 4 == 2:
        #     problem_4 = sample_problem_3(**kws, trialNumber=trialNumber)
        #     main.append(problem_4)
        # elif trialNumber % 4 == 3:
        #     problem_5 = sample_problem_4(**kws, trialNumber=trialNumber)
        #     main.append(problem_5)
        trialNumber += 1


    random.shuffle(main)
    learn_rewards = {'trial_sets': [trial_sets]}
    practice = sample_practice(**kws)
    practice_revealed = [sample_problem(**kws)]
    intro_hover = sample_problem(**kws)
    practice_hover = [sample_problem(**kws)]
    intro = intro_problem(**kws, rewards=[None] * n)
    collect_all = intro_problem(**kws, rewards= [1,2,4,3,0,-1,-2,-3,-4,0])
    

    return {
        'intro': intro,
        'collect_all': collect_all,
        'learn_rewards': learn_rewards,
        'practice': practice,
        'main': main
    }
    
def reward_info(invert=False):
    reward_info = {}
    png = [
        "pattern_1", "pattern_2", "pattern_3", "pattern_4", "pattern_5",
        "pattern_6", "pattern_7", "pattern_8", "pattern_9"
    ]
    images = [
        "images[0]", "images[1]", "images[2]", "images[3]", "images[4]",
        "images[5]", "images[6]", "images[7]", "images[8]"
    ]

    rewards = list(range(-4, 5))  # This loops from -4 to 4 inclusive

    if invert:
        rewards = [-x for x in rewards]  # Invert the sign of each reward

    for index, reward in enumerate(rewards):
        desc = png[index]  
        image_path = images[index] 
        reward_info[reward] = {"desc": desc, "image": image_path}
    
    return reward_info

def reward_contours(n=9):
    png = ["pattern_1","pattern_2", "pattern_3","pattern_4", "pattern_5","pattern_6", "pattern_7", "pattern_8", "pattern_9"]
    if len(png) < n:
        png.extend(["pattern_default"] * (n - len(png)))

    # return dict(zip([4,3,2,1,0,-1,-2,-3,-4], png)) 
    return dict(zip([-4,-3,-2,-1,0,1,2,3,4], png))

from random import sample

def reward_graphics(n = 9,rewards = [4,3,2,1,0,-1,-2,-3,-4]):
    emojis = [
        '🎈','🔑','🎀','🎁','📌','✏️','🔮','💰','⚙️','💎','💡','⏰',
        '✈️','🍫','🍎','🧀','🍪','🌞','⛄️', '🐒','👑','👟', '🤖','🤡',
    ]
    fixed_rewards = [str(i) for i in rewards]  # convert numbers to strings
    return dict(zip(fixed_rewards, sample(emojis, n)))

def circle_layout(N):
    angles = np.pi/2 + np.arange(0, N) * 2 * np.pi / N  # calculate angles for each point
    x = (np.cos(angles) + 1) / 2 - 0.5
    y = (np.sin(angles) + 1) / 2 - 0.5
    return [(-xi, yi) for xi, yi in zip(x, y)]

# Generate trials
subj_trials = [make_trials() for _ in range(2)]

# Directory setup
dest = "config/m2"
os.makedirs(dest, exist_ok=True)

# Save trials as JSON
for i, trials in enumerate(subj_trials, start=0):
    parameters = {
        'reward_info': reward_info(), 
        'images': IMAGES,  # Map permutation indices to image filenames
        'points_per_cent': 1,
        'revealed': True,
        'layout': circle_layout(10)
    }
    with open(f"{dest}/{i}.json", "w", encoding='utf-8') as file:
        json.dump({"parameters": parameters, "trials": trials}, file, ensure_ascii=False)

n = 10
rewards = [-1,-2 , 0, 1, 2]
print(sample_problem_4(n=10))


# Example usage
# trials = make_trials()
# print(trials)


# # Example usage
# print("regular tree:", regular_tree([2, 2]))  # Example of regular tree with specific branching
