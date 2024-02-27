import numpy as np
from ether.util import harmonic_random_number, topsis, is_server_node

def get_potential_targets_randomly(sorted_nodes, node, num_nodes, selection_size_factor, max_num_links, servers_max_num_links):
    """
    Select potential targets randomly.
    """
    potential_targets = []
    found_targets = 0
    attempt_counter = 0
    max_attempts = num_nodes * 10  # Adjusted to limit attempts for each iteration

    while found_targets < selection_size_factor and attempt_counter < max_attempts:
        attempt_counter += 1
        index = harmonic_random_number(num_nodes) - 1
        potential_target = sorted_nodes[index]

        if is_server_node(potential_target):
            if (len(potential_target.long_distance_links) < servers_max_num_links and
                potential_target != node and
                potential_target not in node.successor_links and
                potential_target not in node.predecessor_links and
                potential_target not in node.long_distance_links and
                potential_target not in potential_targets):
                potential_targets.append(potential_target)
                found_targets += 1
        elif (len(potential_target.long_distance_links) < max_num_links and
            potential_target != node and
            potential_target not in node.successor_links and
            potential_target not in node.predecessor_links and
            potential_target not in node.long_distance_links and
            potential_target not in potential_targets):
            potential_targets.append(potential_target)
            found_targets += 1

    print(f"potential_targets for {node} {potential_targets}")
    return potential_targets


def get_potential_targets_from_neighborhood(sorted_nodes, node, num_nodes, selection_size_factor, max_num_links, servers_max_num_links):
    """
    Select potential targets from the neighborhood.
    """
    potential_targets = []
    found_targets = 0
    attempt_counter = 0
    max_attempts = num_nodes * 10  # Adjusted to limit attempts for each iteration

    # Try to find one potential target
    while not found_targets and attempt_counter < max_attempts:
        attempt_counter += 1
        # Select a random target based on harmonic distribution
        index = harmonic_random_number(num_nodes) - 1
        potential_target = sorted_nodes[index]

        if is_server_node(potential_target):
            if (len(potential_target.long_distance_links) < servers_max_num_links and
                potential_target != node and
                potential_target not in node.successor_links and
                potential_target not in node.predecessor_links and
                potential_target not in node.long_distance_links and
                potential_target not in potential_targets):
                potential_targets.append(potential_target)
                found_targets += 1
        elif (len(potential_target.long_distance_links) < max_num_links and
            potential_target != node and
            potential_target not in node.successor_links and
            potential_target not in node.predecessor_links and
            potential_target not in node.long_distance_links and
            potential_target not in potential_targets):
            potential_targets.append(potential_target)
            found_targets += 1

    print(f"potential_targets so far{potential_targets}")
    # Find more targets before and after the found target
    if found_targets > 0:
        search_radius = int(selection_size_factor / 2)
        print(f"search_radius {search_radius}")
        find_additional_targets(sorted_nodes, node, index, -1, search_radius, max_num_links, servers_max_num_links, potential_targets)  # Search backwards for new targets
        find_additional_targets(sorted_nodes, node, index, 1, search_radius, max_num_links, servers_max_num_links, potential_targets)   # Search forwards for new targets

    print(f"potential_targets for {node} {potential_targets}")
    return potential_targets


def topsis(criteria_matrix, weights, is_benefit):
    """
    Perform the TOPSIS decision-making method on the given criteria matrix.
    """
    # Ensure the matrix is not empty
    if criteria_matrix.size == 0:
        raise ValueError("Input matrix is empty")

    # Calculate the Euclidean norm of each column
    column_norms = np.sqrt((criteria_matrix ** 2).sum(axis=0))

    # Normalize the matrix, setting columns with zero norm to zero
    nonzero_columns = column_norms != 0
    norm_matrix = np.zeros_like(criteria_matrix)
    norm_matrix[:, nonzero_columns] = criteria_matrix[:, nonzero_columns] / column_norms[nonzero_columns]

    # Weighted normalized matrix
    weighted_matrix = norm_matrix * weights

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
    scores = np.divide(dist_worst, dist_best + dist_worst, out=np.zeros_like(dist_worst), where=(dist_best + dist_worst) != 0)

    return scores


def decide_topsis(criteria_matrix, weights, is_benefit):
    """
    Decide the best target using the TOPSIS method.
    """
    scores = topsis(criteria_matrix, weights, is_benefit)
    print(f"scores {scores}")
    best_target_index = np.argmax(scores)
    return best_target_index


def find_additional_targets(sorted_nodes, node, start_index, direction, max_count, max_num_links, servers_max_num_links, potential_targets):
    """
    Find additional potential targets in a specified direction.
    """
    current_index = start_index
    count = 0

    while count < max_count:
        current_index = (current_index + direction) % len(sorted_nodes)
        if current_index == start_index:
            break  # Avoid infinite loop by stopping if we've looped around

        new_target = sorted_nodes[current_index]
        if new_target == node:
            continue  # Skip if it's the node itself

        valid_target_conditions = (
            len(new_target.long_distance_links) < max_num_links,
            new_target not in node.successor_links,
            new_target not in node.predecessor_links,
            new_target not in node.long_distance_links,
            new_target not in potential_targets
        )

        if all(valid_target_conditions):
            server_condition = is_server_node(new_target) and len(new_target.long_distance_links) < servers_max_num_links
            if server_condition or not is_server_node(new_target):
                potential_targets.append(new_target)
                count += 1

    return potential_targets