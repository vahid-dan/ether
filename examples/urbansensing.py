import math
import random
import numpy as np
import srds
import datetime
import os
import csv
import statistics

import ether.blocks.nodes as nodes
from ether.core import Node
from ether.vis import visualize_symphony_structure, visualize_topology, print_symphony_structure
from ether.overlay import SymphonyOverlay
from examples.vivaldi.util import execute_vivaldi
from ether.util import print_location_ids, print_cell_costs, generate_topology, calculate_total_latency, calculate_total_cell_cost


def main(link_selection_method='topsis',
         weights=[1, 1],
         candidate_list_size_factor=2,
         num_neighborhoods=3,
         num_nodes_per_neighborhood=5,
         num_cloudlets=2,
         num_racks=2,
         num_servers_per_rack=4,
         node_type=nodes.rpi4,
         density_params=(0.82, 2.02),
         metered_edge_nodes_percentage=50,
         num_pairs=10,
         results_dir="results"):

    topology = generate_topology(num_neighborhoods,
                                 num_nodes_per_neighborhood,
                                 num_cloudlets,
                                 num_racks,
                                 num_servers_per_rack,
                                 node_type,
                                 density_params)

    # Update Vivaldi coordinates based on network interactions for all nodes
    execute_vivaldi(topology, node_filter=lambda n: isinstance(n, Node), min_executions=300)

    overlay_nodes = topology.get_nodes()
    # random_pairs = random.sample(overlay_nodes, 2 * num_pairs)  # Pairs of source and destination nodes
    random_pairs = []
    for _ in range(num_pairs):
        source_node = random.choice(overlay_nodes)
        destination_node = random.choice(overlay_nodes)
        random_pairs.append((source_node, destination_node))

    print(f"random_pairs {random_pairs}")
    results = {}

    total_latencies = []
    total_cell_costs = []

    # Create a dictionary to map pairs to source and destination node names
    pair_to_nodes = {}

    num_nodes = len(overlay_nodes) # Number of nodes
    max_num_links = round(math.log(num_nodes)) # Max number of long-distance links per node

    print(f'Number of Nodes: {num_nodes}')

    # Format topology parameters into a string for the filename
    topology_params_str = f"{link_selection_method}_w{weights[0]}-{weights[1]}_clsf{candidate_list_size_factor}_nn{num_neighborhoods}_npn{num_nodes_per_neighborhood}_nc{num_cloudlets}_nr{num_racks}_nspr{num_servers_per_rack}_nt{node_type.__name__}_dp{density_params[0]}-{density_params[1]}_menp{metered_edge_nodes_percentage}"

    # Initialize the Symphony overlay with these nodes
    symphony_overlay = SymphonyOverlay(overlay_nodes, seed=SEED)
    symphony_overlay.assign_cell_costs(metered_edge_nodes_percentage)
    symphony_overlay.set_successor_links()
    symphony_overlay.set_long_distance_links(topology=topology,
                                             max_num_links=max_num_links,
                                             link_selection_method=link_selection_method,
                                             candidate_list_size_factor=candidate_list_size_factor,
                                             weights=weights,
                                             is_benefit=[False, False])

    # print_cell_costs(overlay_nodes)

    # print_location_ids(overlay_nodes)

    # visualize_topology(topology)

    # print_symphony_structure(symphony_overlay)

    # visualize_symphony_structure(topology)

    # for i in range(num_pairs):
    #     source_node = random_pairs[i]
    #     destination_node = random_pairs[i + 1]

    #     # Store the source and destination node names in the dictionary
    #     pair_to_nodes[f"{i+1}"] = (source_node.name, destination_node.name)

    #     symphony_path = symphony_overlay.find_symphony_path(source_node, destination_node)

    #     print(f"symphony_path {symphony_path}")

    #     total_latency = calculate_total_latency(symphony_path, topology)
    #     total_cell_cost = calculate_total_cell_cost(symphony_path)
    #     total_latencies.append(total_latency)
    #     total_cell_costs.append(total_cell_cost)

    #     results[f"{i+1}"] = {"latency": total_latency, "cell_cost": total_cell_cost}
    for i, (source_node, destination_node) in enumerate(random_pairs, start=1):
        # Assuming source_node and destination_node have an attribute 'name'
        pair_to_nodes[str(i)] = (source_node.name, destination_node.name)

        symphony_path = symphony_overlay.find_symphony_path(source_node, destination_node)
        print(f"symphony_path {symphony_path}")

        total_latency = calculate_total_latency(symphony_path, topology)
        total_cell_cost = calculate_total_cell_cost(symphony_path)
        
        total_latencies.append(total_latency)
        total_cell_costs.append(total_cell_cost)

        results[str(i)] = {"latency": total_latency, "cell_cost": total_cell_cost}
    # Filename for results
    filename = f'{topology_params_str}_pairs{num_pairs}_seed{SEED}'
    with open(f'{results_dir}/{filename}.csv', mode='w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['ID', 'Source', 'Destination', 'Latency', 'Cell Cost'])
        for pair, results in results.items():
            source_name, destination_name = pair_to_nodes[pair]
            latency = results['latency']  # Access latency from the dictionary
            cell_cost = results['cell_cost']  # Access cell cost from the dictionary
            writer.writerow([pair, source_name, destination_name, latency, cell_cost])

    # Calculate statistics for TOPSIS distances
    total_latency_avg = statistics.mean(total_latencies)
    total_latency_median = statistics.median(total_latencies)
    total_latency_min = min(total_latencies)
    total_latency_max = max(total_latencies)
    total_latency_std_dev = statistics.stdev(total_latencies)

    total_cell_cost_avg = statistics.mean(total_cell_costs)
    total_cell_cost_median = statistics.median(total_cell_costs)
    total_cell_cost_min = min(total_cell_costs)
    total_cell_cost_max = max(total_cell_costs)
    total_cell_cost_std_dev = statistics.stdev(total_cell_costs)

    # Store statistics in separate CSV files
    with open(f'{results_dir}/{filename}_stats.csv', mode='w', newline='') as random_stats_file:
        writer = csv.writer(random_stats_file)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Latency Average', total_latency_avg])
        writer.writerow(['Latency Median', total_latency_median])
        writer.writerow(['Latency Minimum', total_latency_min])
        writer.writerow(['Latency Maximum', total_latency_max])
        writer.writerow(['Latency Standard Deviation', total_latency_std_dev])
        writer.writerow(['Cell Cost Average', total_cell_cost_avg])
        writer.writerow(['Cell Cost Median', total_cell_cost_median])
        writer.writerow(['Cell Cost Minimum', total_cell_cost_min])
        writer.writerow(['Cell Cost Maximum', total_cell_cost_max])
        writer.writerow(['Cell Cost Standard Deviation', total_cell_cost_std_dev])

if __name__ == '__main__':
    num_neighborhoods = 2
    num_nodes_per_neighborhood = 10
    num_cloudlets = 3
    num_racks = 4
    num_servers_per_rack = 4
    node_type = nodes.rpi4
    density_params = (0.82, 2.02)
    metered_edge_nodes_percentage = 80
    num_pairs = 20
    candidate_list_size_factor = 2
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    link_selection_method = "topsis"
    weights = [1, 1]
    SEED = 17
    random.seed(SEED)
    srds.seed(SEED)
    np.random.seed(SEED)
    main(link_selection_method,
         weights,
         candidate_list_size_factor,
         num_neighborhoods,
         num_nodes_per_neighborhood,
         num_cloudlets,
         num_racks,
         num_servers_per_rack,
         node_type,
         density_params,
         metered_edge_nodes_percentage,
         num_pairs,
         results_dir)
    # for SEED in range(200, 202):
    #     for link_selection_method in ['topsis', 'random']:  # Loop for both link selection methods
    #         print(f"Running simulation with SEED={SEED} and link_selection_method={link_selection_method}")
    #         # Set the seed for reproducibility
    #         random.seed(SEED)
    #         srds.seed(SEED)
    #         np.random.seed(SEED)
    #         # Call the main function with the current SEED and link_selection_method
    #         main(link_selection_method,
    #              candidate_list_size_factor,
    #              num_neighborhoods,
    #              num_nodes_per_neighborhood,
    #              num_cloudlets,
    #              num_racks,
    #              num_servers_per_rack,
    #              node_type,
    #              density_params,
    #              metered_edge_nodes_percentage,
    #              num_pairs,
    #              results_dir)