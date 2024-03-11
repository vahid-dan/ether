import random
import numpy as np
import pandas as pd
import srds
import os
import csv
import statistics

import ether.blocks.nodes as nodes
from ether.core import Node
from ether.vis import visualize_symphony_structure, visualize_topology, print_symphony_structure
from ether.overlay import SymphonyOverlay
from examples.vivaldi.util import execute_vivaldi
from ether.util import generate_topology, calculate_total_latency, calculate_total_cell_cost
from ether.simulation import NetworkSimulation


def main(target_selection_strategy='harmonic',
         decision_method='topsis',
         weights=np.array([1, 1]),
         num_neighborhoods=3,
         num_nodes_per_neighborhood=5,
         num_cloudlets=2,
         num_racks=2,
         num_servers_per_rack=4,
         workload_quota=100,
         num_cloud_racks=1,
         num_cloud_servers_per_rack=1,
         node_types_and_shares=[(nodes.rpi4, 95), (nodes.rpi3, 5)],
         density_params=(0.82, 2.02),
         metered_edge_nodes_percentage=50,
         num_pairs=10,
         results_dir="results"):

    topology = generate_topology(num_neighborhoods,
                                 num_nodes_per_neighborhood,
                                 num_cloudlets,
                                 num_racks,
                                 num_servers_per_rack,
                                 workload_quota,
                                 num_cloud_racks,
                                 num_cloud_servers_per_rack,
                                 node_types_and_shares,
                                 density_params)

    # Update Vivaldi coordinates based on network interactions for all nodes
    execute_vivaldi(topology, node_filter=lambda n: isinstance(n, Node), min_executions=300)

    all_nodes = topology.get_nodes()
    num_all_nodes = len(all_nodes)
    print(f'Number of All Nodes: {num_all_nodes}')

    # Format topology parameters into a string for the filename
    topology_params_str = f"{target_selection_strategy}_{decision_method}_w{weights[0]}-{weights[1]}_nn{num_neighborhoods}_npn{num_nodes_per_neighborhood}_nc{num_cloudlets}_nr{num_racks}_nspr{num_servers_per_rack}_dp{density_params[0]}-{density_params[1]}_menp{metered_edge_nodes_percentage}"

    # Initialize the Symphony overlay with these nodes
    symphony_overlay = SymphonyOverlay(all_nodes, seed=SEED)

    symphony_overlay.assign_cell_costs(metered_edge_nodes_percentage)
    symphony_overlay.set_successor_links()
    symphony_overlay.set_long_distance_links(topology=topology,
                                             target_selection_strategy=target_selection_strategy,
                                             decision_method=decision_method,
                                             weights=weights,
                                             is_benefit=np.array([False, False]))
    # visualize_symphony_structure(topology)

    symphony_overlay.remove_links_from_pendant_nodes()
    symphony_overlay.set_successor_links()
    symphony_overlay.remove_overlapping_long_distance_links()
    symphony_overlay.set_long_distance_links(topology=topology,
                                             target_selection_strategy=target_selection_strategy,
                                             decision_method=decision_method,
                                             weights=weights,
                                             is_benefit=np.array([False, False]))
    symphony_overlay.set_bridge_links(topology=topology,
                                      weights=weights,
                                      is_benefit=np.array([False, False]))

    visualize_topology(topology)
    # print_symphony_structure(symphony_overlay)
    # print(f"path: {symphony_overlay.find_symphony_path(all_nodes[0], all_nodes[1])}")
    visualize_symphony_structure(topology)

    random_pairs = []
    for _ in range(num_pairs):
        source_node = random.choice(all_nodes)
        print(f"source_node.capacity {source_node.capacity}")
        print(f"source_node.processing_power {source_node.processing_power}")
        print(f"source_node.workload_quota {getattr(source_node, 'workload_quota', 0)}")
        print(f"source_node.location_id {getattr(source_node, 'location_id', None)}")
        destination_node = random.choice(all_nodes)
        random_pairs.append((source_node, destination_node))

    results = {}
    total_latencies = []
    total_cell_costs = []
    pair_to_nodes = {}  # Create a dictionary to map pairs to source and destination node names

    for i, (source_node, destination_node) in enumerate(random_pairs, start=1):
        pair_to_nodes[str(i)] = (source_node.name, destination_node.name)

        symphony_path = symphony_overlay.find_symphony_path(source_node, destination_node)

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

    print(f"Randomness Check {random.random()}")

    simulation = NetworkSimulation(symphony_overlay)
    simulation.simulate_application_traffic()
    simulation.print_traffic_matrix()

if __name__ == '__main__':
    num_neighborhoods = 2
    num_nodes_per_neighborhood = 4
    num_cloudlets = 3
    num_racks = 1
    num_servers_per_rack = 2
    workload_quota = 150
    num_cloud_racks=1
    num_cloud_servers_per_rack=1
    node_types_and_shares = [
        (nodes.rpi4, 90),
        (nodes.rpi3, 10)
    ]
    density_params = (0.82, 2.02)
    metered_edge_nodes_percentage = 40
    num_pairs = 100
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    target_selection_strategy = 'harmonic'
    decision_method = "topsis"
    weights = np.array([1, 1])
    SEED = 2
    random.seed(SEED)
    srds.seed(SEED)
    np.random.seed(SEED)
    main(target_selection_strategy,
         decision_method,
         weights,
         num_neighborhoods,
         num_nodes_per_neighborhood,
         num_cloudlets,
         num_racks,
         num_servers_per_rack,
         workload_quota,
         num_cloud_racks,
         num_cloud_servers_per_rack,
         node_types_and_shares,
         density_params,
         metered_edge_nodes_percentage,
         num_pairs,
         results_dir)
