import re
import numpy as np
import random

from ether.core import Node

__size_conversions = {
    'K': 10 ** 3,
    'M': 10 ** 6,
    'G': 10 ** 9,
    'T': 10 ** 12,
    'P': 10 ** 15,
    'E': 10 ** 18,
    'Ki': 2 ** 10,
    'Mi': 2 ** 20,
    'Gi': 2 ** 30,
    'Ti': 2 ** 40,
    'Pi': 2 ** 50,
    'Ei': 2 ** 60
}

__size_pattern = re.compile(r"([0-9]+)([a-zA-Z]*)")


def parse_size_string(size_string: str) -> int:
    m = __size_pattern.match(size_string)
    if len(m.groups()) > 1:
        number = m.group(1)
        unit = m.group(2)
        return int(number) * __size_conversions.get(unit, 1)
    else:
        return int(m.group(1))


def to_size_string(num_bytes, unit='M', precision=1) -> str:
    factor = __size_conversions[unit]
    value = num_bytes / factor

    fmt = f'%0.{precision}f{unit}'

    return fmt % value


def harmonic_random_number(num_nodes):
    # Generate the harmonic series
    harmonic_series = np.array([1.0 / (i + 1) for i in range(1, num_nodes + 1)])
    
    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(harmonic_series) / np.sum(harmonic_series)
    
    # Generate a uniform random number
    random_num = np.random.uniform()
    
    # Find the index where this random number would fit in the CDF
    index = np.searchsorted(cdf, random_num)
    
    # Return the value corresponding to this index
    return index + 1  # Adding 1 because index starts from 0


def topsis(matrix, weights, is_benefit):
    # Ensure the matrix is not empty
    if matrix.size == 0:
        raise ValueError("Input matrix is empty")

    # Calculate the Euclidean norm of each column
    column_norms = np.sqrt((matrix ** 2).sum(axis=0))

    # Initialize the normalized matrix
    norm_matrix = np.zeros_like(matrix)

    # Normalize the matrix, setting columns with zero norm to zero
    nonzero_columns = column_norms != 0
    norm_matrix[:, nonzero_columns] = matrix[:, nonzero_columns] / column_norms[nonzero_columns]

    # Weighted normalized matrix
    weighted_matrix = norm_matrix * weights

    # Ensure weighted_matrix is not empty
    if weighted_matrix.size == 0:
        raise ValueError("Weighted matrix is empty")

    # Initialize ideal best and worst arrays
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    # Determine the ideal best and worst for each criterion
    for i in range(weighted_matrix.shape[1]):
        if is_benefit[i]:
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    # Calculate the distances to the ideal best and ideal worst
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # Calculate the performance score, avoiding division by zero
    score = np.divide(dist_worst, dist_best + dist_worst, out=np.zeros_like(dist_worst), where=(dist_best + dist_worst) != 0)

    return score

    
# Function to calculate total latency for a path
def calculate_total_latency(path, topology):
    total_latency = 0
    for i in range(len(path) - 1):
        total_latency += topology.latency(path[i], path[i + 1], use_coordinates=False)
    return total_latency


# Function to calculate total cell_cost for a path
def calculate_total_cell_cost(path):
    total_cell_cost = 0
    incoming_traffic_cost = 1
    outgoing_traffic_cost = 1
    for i in range(len(path)):
        # Skip the incoming traffic cost for the first node
        if i > 0:
            if path[i].name.startswith('rpi4'):
                total_cell_cost += incoming_traffic_cost

        # Skip the outgoing cost for the last node
        if i < len(path) - 1:
            if path[i].name.startswith('rpi4'):
                total_cell_cost += outgoing_traffic_cost

    return total_cell_cost


# Function to assign random cell_cost for a specific percentage of the edge nodes
def assign_cell_cost(targets, percentage):
    # Filter targets that are edge nodes
    edge_targets = [target for target in targets if is_edge_node(target)]
    
    # Calculate number of nodes to assign random cost
    num_random_cost = int(len(edge_targets) * (percentage / 100.0))
    
    # Randomly select nodes for random cost assignment
    random_cost_targets = random.sample(edge_targets, num_random_cost) if edge_targets else []
    
    # Initialize dictionary to store cell costs
    cell_costs = {}
    
    # Assign random cost to selected nodes and 0 to others
    for target in targets:
        if target in random_cost_targets:
            cell_costs[target.name] = random.uniform(0, 1)
        elif is_edge_node(target):
            cell_costs[target.name] = 0
        else:
            # Assign 0 or maintain current cost for non-edge nodes
            cell_costs[target.name] = 0
    
    return cell_costs


def print_cell_cost(nodes):
    for node in nodes:
        if is_edge_node(node):
            print(f"{node.name} is an edge node.")
            cell_cost = getattr(node, 'cell_cost', None)
            if cell_cost:
                print(f"cell_cost {cell_cost}")    
        elif is_server_node(node):
            print(f"{node.name} is a server node.")


def print_location_ids(nodes):
    for node in nodes:
        if is_server_node(node):
            print(f"{node.name} is a server node.")
            location_id = getattr(node, 'location_id', None)
            if location_id:
                print(f"location_id {location_id}")
        elif is_edge_node(node):
            print(f"{node.name} is an edge node.")


def is_server_node(node: Node) -> bool:
    return node.labels.get('ether.edgerun.io/type') == 'server'


def is_edge_node(node: Node) -> bool:
    edge_types = {'sbc', 'embai'}
    return node.labels.get('ether.edgerun.io/type') in edge_types
