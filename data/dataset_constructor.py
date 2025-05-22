###
###
###


import numpy as np
import json
import networkx as nx
import argparse
import os
import subprocess
import tempfile
import random
import math
from itertools import combinations 
from tqdm import tqdm


# HELPERS
# ------------------------------------------------------------------------------------
def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def tour_cost(instance, tour):
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance(instance[tour[i]], instance[tour[(i+1) % num_cities]])
    return cost


def generate_tsp_instance(num_cities, low=0.0, high=1.0):
    """
    Generate a TSP instance: array of shapes (num_cities, 2) with random coordinates.
    """
    return np.random.uniform(low, high, size=(num_cities, 2))


def rotate_tour_to_start_with_zero(tour):
    if 0 in tour:
        i = tour.index(0)
        return tour[i:] + tour[:i]
    return tour


# SOLVERS
# ------------------------------------------------------------------------------------
def random_tsp(instance):
    instance_range = ([i for i in range(1, len(instance))])
    random.shuffle(instance_range)
    return [0] + instance_range


def greedy_tsp(instance):
    num_cities = instance.shape[0]
    start = 0
    tour = [start]
    unvisited = set(range(num_cities))
    unvisited.remove(start)
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance(instance[current], instance[city]))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return tour


def two_opt_tsp(instance, initial_tour=None):
    if initial_tour is None:
        tour = greedy_tsp(instance)
    else:
        tour = initial_tour[:]
    improvement = True
    best_cost = tour_cost(instance, tour)
    
    while improvement:
        improvement = False
        n = len(tour)
        for i in range(1, n - 1):
            for k in range(i+1, n):
                new_tour = two_opt_swap(tour, i, k)
                new_cost = tour_cost(instance, new_tour)
                if new_cost < best_cost:
                    tour = new_tour
                    best_cost = new_cost
                    improvement = True
                    break
            if improvement:
                break
    return tour


def two_opt_swap(tour, i, k):
    """Performs a 2‑opt swap, reverses the order of the tour elements between indices i and k"""
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour


def christofides_tsp(instance):
    num_cities = instance.shape[0]
    G = nx.Graph()

    for i in range(num_cities):
        G.add_node(i)

    for i in range(num_cities):
        for j in range(i+1, num_cities):
            G.add_edge(i, j, weight=distance(instance[i], instance[j]))

    T = nx.minimum_spanning_tree(G, weight='weight')

    odd_degree_nodes = [node for node in T.nodes() if T.degree(node) % 2 == 1]

    odd_graph = nx.Graph()
    for i in range(len(odd_degree_nodes)):
        for j in range(i+1, len(odd_degree_nodes)):
            u = odd_degree_nodes[i]
            v = odd_degree_nodes[j]
            odd_graph.add_edge(u, v, weight=distance(instance[u], instance[v]))

    matching = nx.algorithms.matching.min_weight_matching(odd_graph, weight='weight')

    multi_graph = nx.MultiGraph(T)
    for u, v in matching:
        multi_graph.add_edge(u, v, weight=distance(instance[u], instance[v]))

    circuit = list(nx.eulerian_circuit(multi_graph, source=0))

    visited = set()
    tour = []
    for u, v in circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)

    if tour[0] != tour[-1]:
        tour.append(tour[0])

    tour = tour[:-1]
    return tour


def heldkarp_tsp(coords):
    """
    Solve the TSP optimally using the Held-Karp dynamic programming algorithm.

    Parameters
    ----------
    coords : sequence of (x, y) pairs
        The coordinates of n cities (n should be small, e.g., n <= 15).

    Returns
    -------
    best_cost : float
        The length of the optimal tour.
    best_tour : list of int
        The optimal cycle, starting and ending at city 0, e.g., [0, ..., 0].
    """
    n = len(coords)
    if n == 0:
        return 0.0, []

    # Precompute Euclidean distance matrix
    dist = [[math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]

    # DP table and parent pointers
    dp = {}         # key: (mask, j), value: min cost to reach j using subset mask
    parent = {}     # key: (mask, j), value: previous city index

    # Base cases: distance from start (0) to each j
    for j in range(1, n):
        mask = 1 << (j - 1)
        dp[(mask, j)] = dist[0][j]
        parent[(mask, j)] = 0

    # Fill DP table
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            mask = 0
            for city in subset:
                mask |= 1 << (city - 1)
            for j in subset:
                prev_mask = mask ^ (1 << (j - 1))
                best_cost = float('inf')
                best_prev = None
                for k in subset:
                    if k == j:
                        continue
                    cost = dp[(prev_mask, k)] + dist[k][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = k
                dp[(mask, j)] = best_cost
                parent[(mask, j)] = best_prev

    # Close the tour back to start
    full_mask = (1 << (n - 1)) - 1
    best_cost = float('inf')
    best_last = None
    for j in range(1, n):
        cost = dp[(full_mask, j)] + dist[j][0]
        if cost < best_cost:
            best_cost = cost
            best_last = j

    # Reconstruct optimal path
    tour = [0]
    mask = full_mask
    last = best_last
    stack = []
    for _ in range(n - 1):
        stack.append(last)
        prev = parent[(mask, last)]
        mask ^= 1 << (last - 1)
        last = prev
    tour += list(reversed(stack))
    tour.append(0)

    return best_cost, tour


# LKH
# ------------------------------------------------------------------------------------
def write_tsplib_instance(instance, filename, scale=10000):
    """Save the instance in a TSPLIB format"""

    int_coords = (instance * scale).astype(int)
    n = int_coords.shape[0]

    with open(filename, "w") as f:
        f.write("NAME: temp_instance\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(int_coords, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")


def write_lkh_parameter_file(tsp_filename, par_filename, tour_filename):
    with open(par_filename, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_filename}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_filename}\n")
        #f.write(f"SEED = {random.randint(1, 1_000_000)}\n")
        f.write("RUNS = 20\n")
        #sf.write("MAX_TRIALS = 10000")
        f.write("MOVE_TYPE = 5\n")
        f.write("PATCHING_C = 3\n")
        f.write("PATCHING_A = 2\n")
        f.write("TRACE_LEVEL = 0\n")


def parse_lkh_tour(filename):
    tour = []
    reading = False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                reading = True
                continue
            if reading:
                if line in ("-1", "EOF"):
                    break
                try:
                    tour.append(int(line) - 1)
                except ValueError:
                    pass

    i0 = tour.index(0)
    cycle = tour[i0:] + tour[:i0]
    return cycle[1:]


def lkh_tsp(instance, lkh_path="lkh/LKH-3.0.13/LKH"):
    """Use the LKH solver to solve the TSP instance"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tsp_filename = os.path.join(tmpdir, "temp_instance.tsp")
        par_filename = os.path.join(tmpdir, "lkh.par")
        tour_filename = os.path.join(tmpdir, "temp_instance.tour")
        write_tsplib_instance(instance, tsp_filename)
        write_lkh_parameter_file(tsp_filename, par_filename, tour_filename)

        try:
            subprocess.run([lkh_path, par_filename], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(e)
            return None

        if os.path.exists(tour_filename):
            tour = parse_lkh_tour(tour_filename)
            return tour
        else:
            return None


# DATASET
# ------------------------------------------------------------------------------------
def generate_dataset(num_instances, num_cities, teachers, weights=None):
    solvers = {
        'random': lambda inst: rotate_tour_to_start_with_zero(random_tsp(inst)),
        'greedy': greedy_tsp,
        '2-opt': two_opt_tsp,
        'christofides': christofides_tsp,
        'lkh': lkh_tsp
    }

    dataset = []
    teacher_list = teachers if teachers != 'mixed' else list(weights.keys())

    for _ in tqdm(range(num_instances), desc=f"GEN {num_instances}x{num_cities}, teachers={teachers}"):
        instance = generate_tsp_instance(num_cities)

        if teachers == 'mixed':
            t = random.choices(teacher_list, weights=[weights[t] for t in teacher_list])[0]
        else:
            t = teachers[0]

        tour = solvers[t](instance)
        if tour is None:
            continue

        full_cost = tour_cost(instance, tour + [tour[0]])
        
        masks = []
        actions = []
        rtgs = []

        remaining = -full_cost
        unvisited = set(range(num_cities))
        unvisited.remove(0)
        prev = 0

        for nxt in tour[1:]:
            reward = -distance(instance[prev], instance[nxt])
            remaining -= reward

            mask = [1 if city in unvisited else 0 for city in range(num_cities)]
            mask = mask[1:]

            masks.append(mask)
            actions.append(nxt)
            rtgs.append(remaining)

            unvisited.remove(nxt)
            prev = nxt

        dataset.append({
            'x': instance.tolist(),
            'masks': masks,
            'actions': actions,
            'rtgs': rtgs,
        })

    return dataset


def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate TSP imitation dataset with returns-to-go")
    parser.add_argument('--num_instances', type=int, default=1000000)
    parser.add_argument('--num_cities', type=int, default=10)
    parser.add_argument('--teachers', type=str, default='mixed')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--output', type=str, default='MIXED10_TRAIN1M.json')
    args = parser.parse_args()
    
    if args.teachers != 'mixed':
        teacher_list = [t.strip() for t in args.teachers.split(',')]
        weights = None
    else:
        teacher_list = 'mixed'
        if args.weights is None:
            parser.error("--weights REQUIRED WHEN --teachers=mixed")
        weights = {}
        for pair in args.weights.split(','):
            name,val = pair.replace(':','=').split('=')
            weights[name.strip()] = float(val)
    ds = generate_dataset(args.num_instances, args.num_cities, teacher_list, weights)
    
    save_dataset(ds, args.output)
    
    print(f"DATASET SAVED TO {args.output}")
